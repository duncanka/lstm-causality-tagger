#!/usr/bin/env python
# NOTE: This script must be run with Causeway and NLPypline on the PYTHONPATH.

from __future__ import absolute_import, print_function

from collections import defaultdict, deque
from copy import copy
from gflags import FLAGS
import logging
from nltk.util import flatten
from os.path import splitext
import sys

from causeway.because_data import CausalityStandoffReader, CausationInstance
from nlpypline.data.io import DirectoryReader, InstancesDocumentWriter


class CausalityOracleTransitionWriter(InstancesDocumentWriter):
    def _write_instance(self, document, sentence):
        tokens = [token for token in sentence.tokens[1:]] # skip ROOT

        # Print sentence-initial line with tokens and POS tags.
        print(u', '.join(u'/'.join([t.original_text.replace(' ', ''), t.pos])
                         for t in tokens),
              file=self._file_stream)

        # Initialize state. lambda_1 is unexamined tokens to the left of the
        # current token; lambda_2 is examined tokens to the left; and likewise
        # for lambda_4 and lambda_3, respectively, to the right.
        self.lambda_1 = []
        self.lambda_2 = deque() # we'll be appending to the left end
        self.lambda_3 = []
        # We'll be moving stuff off of the left end of lambda_4.
        self.lambda_4 = deque(tokens)
        self.lambdas = [self.lambda_1, self.lambda_2, self.lambda_3,
                        self.lambda_4]
        self.rels = []
        self.extrasentential_instances = 0
        self.extrasentential_args = []
        self._last_op = None

        connectives_to_instances = defaultdict(list)
        for causation in sentence.causation_instances:
            # Eliminate any instances with extrasentential tokens, but add them
            # to the count of tokens we can't handle.
            if all([t.parent_sentence is sentence
                    for t in causation.connective]):
                first_conn_token = causation.connective[0]
                connectives_to_instances[first_conn_token].append(causation)
            else:
                self.extrasentential_instances += 1

        # Make sure the instances for each token are sorted by order of
        # appearance.
        for _, causations in connectives_to_instances.iteritems():
            def sort_key(instance):
                return tuple(t.index for t in instance.connective
                             if t.parent_sentence is sentence)
            causations.sort(key=sort_key)

        for current_token in tokens:
            instance_under_construction = None
            token_instances = connectives_to_instances[current_token]
            if token_instances: # some connective starts with this token
                # Record extrasentential argument counts.
                for causation in token_instances:
                    extrasentential_counts = []
                    for arg_type in causation.get_arg_types():
                        arg_extrasentential_count = 0
                        arg = getattr(causation, arg_type, None)
                        if arg:
                            for token in arg:
                                if token.parent_sentence is not sentence:
                                    arg_extrasentential_count += 1
                        extrasentential_counts.append(arg_extrasentential_count)
                    self.extrasentential_args.append(extrasentential_counts)

                instance_under_construction = self._compare_with_conn(
                    current_token, True, token_instances,
                    instance_under_construction)
                self._compare_with_conn(current_token, False, token_instances,
                                        instance_under_construction)
                self._write_transition(current_token, 'SHIFT')

            else:
                self._write_transition(current_token, 'NO-CONN')

            if current_token is not tokens[-1]:
                self.lambda_1.extend(self.lambda_2)
                self.lambda_2.clear()
                self.lambda_1.append(current_token)
                if self.lambda_3: # we processed some right-side tokens
                    # Skip copy of current token.
                    self.lambda_4.extend(self.lambda_3[1:])
                    # If we didn't use del here, we'd have to reconstruct
                    # self.lambdas.
                    del self.lambda_3[:]
                else: # current_token was a no-conn
                    self.lambda_4.popleft()

        self._write_sentence_footer()
        (self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4,
         self.lambdas, self.rels) = [None] * 6 # Reset; release memory

    def _do_split(self, current_token, last_modified_arg, token_to_compare,
                  instance_under_construction):
        self._write_transition(current_token, 'SPLIT')

        # Figure out where in the connective the token we're replacing is.
        conn_token_index = None
        for i, conn_token in enumerate(instance_under_construction.connective):
            if conn_token.lemma == token_to_compare.lemma:
                conn_token_index = i
        if conn_token_index is None:
            logging.warn("Didn't find a shared word when splitting connective;"
                         " sharing only the first word")
            conn_token_index = 1
        arg_cutoff_index = instance_under_construction.connective[
            conn_token_index].index

        # Don't need a full deep copy...don't want to copy all the sentences and
        # tokens and whatnot. Just make sure we don't modify the original.
        instance_under_construction = copy(instance_under_construction)
        instance_under_construction.__dict__.update(
            {name: copy(arg) for name, arg in
             instance_under_construction.get_named_args().iteritems()})
        instance_under_construction.connective = (
            instance_under_construction.connective[:conn_token_index])
        self.rels.append(instance_under_construction)

        # We need to know which tokens to keep from the argument we were
        # building when we encountered this new connective token. We assume that
        # we should keep any token preceding the connective fragment we're
        # replacing.
        new_argument = [t for t in getattr(instance_under_construction,
                                           last_modified_arg)
                        if t.index < arg_cutoff_index]
        setattr(instance_under_construction, last_modified_arg,
                new_argument)
        instance_under_construction.connective.append(token_to_compare)
        # other_connective_tokens.remove(token_to_compare)
        return instance_under_construction

    def _compare_with_conn(self, current_token, dir_is_left,
                           connective_instances, instance_under_construction):
        if dir_is_left:
            arc_direction = 'LEFT'
            first_uncompared_index = -1
            compared = self.lambda_2
            uncompared = self.lambda_1
        else:
            arc_direction = 'RIGHT'
            first_uncompared_index = 0
            compared = self.lambda_3
            uncompared = self.lambda_4

        conn_instance_index = 0
        conn_instance = connective_instances[conn_instance_index]
        other_connective_tokens = set(flatten(
            [i.connective for i in connective_instances[1:]]))
        other_connective_tokens -= set(conn_instance.connective)
        last_modified_arc_type = None

        def start_new_instance():
            self._write_transition(current_token, "NEW-CONN")
            new_instance = CausationInstance(
                conn_instance.sentence, cause=[], effect=[],
                means=[], connective=[current_token])
            self.rels.append(new_instance)
            return new_instance

        while uncompared:
            token_to_compare = uncompared[first_uncompared_index]

            # First, see if we should split. But don't split on leftward tokens.
            if (not dir_is_left and token_to_compare in other_connective_tokens
                and self._last_op != 'SPLIT'):
                instance_under_construction = self._do_split(
                    current_token, last_modified_arc_type, token_to_compare,
                    instance_under_construction)
                # Move to next
                conn_instance_index += 1
                conn_instance = connective_instances[conn_instance_index]
                # Leave current token to be compared with new connective.
            else:
                # If there's a fragment, record it first, before looking at the
                # args. (The fragment word might still be part of an arg.)
                if (token_to_compare is not current_token
                    and self._last_op not in  # no fragments after splits/frags
                        ['SPLIT',  "CONN-FRAG-{}".format(arc_direction)]
                    and token_to_compare in conn_instance.connective):
                    self._write_transition(current_token,
                                           "CONN-FRAG-{}".format(arc_direction))
                    instance_under_construction.connective.append(
                        token_to_compare)

                arcs_to_add = []
                for arc_type in ['cause', 'effect', 'means']:
                    argument = getattr(conn_instance, arc_type, None)
                    if argument is not None and token_to_compare in argument:
                        arcs_to_add.append(arc_type)
                        # TODO: This will do odd things if there's ever a SPLIT
                        # interacting with a multiple-argument arc.
                        last_modified_arc_type = arc_type
                if arcs_to_add:
                    if instance_under_construction is None:
                        instance_under_construction = start_new_instance()
                    trans = "{}-ARC({})".format(
                        arc_direction, ','.join(arc_type.title()
                                                for arc_type in arcs_to_add))
                    self._write_transition(current_token, trans)
                    for arc_type in arcs_to_add:
                        getattr(instance_under_construction, arc_type).append(
                            token_to_compare)
                else:
                    if instance_under_construction is None:
                        instance_under_construction = start_new_instance()
                    self._write_transition(current_token,
                                           "NO-ARC-{}".format(arc_direction))

                if dir_is_left:
                    compared.appendleft(uncompared.pop())
                else:
                    compared.append(uncompared.popleft())

        return instance_under_construction # make update visible

    def _write_transition(self, current_token, transition):
        stringified_lambdas = [self._stringify_token_list(l)
                               for l in self.lambdas]
        state_line = u"{} {} {token} {} {}".format(
            *stringified_lambdas, token=self._stringify_token(current_token))
        rels_line = self._stringify_rels()
        for line in [state_line, rels_line, unicode(transition)]:
            print(line, file=self._file_stream)
        self._last_op = transition

    def _stringify_token(self, token):
        return u'{}-{}'.format(token.original_text.replace(' ', ''),
                               token.index)

    def _stringify_token_list(self, token_list):
        token_strings = [self._stringify_token(t) for t in token_list]
        return u'[{}]'.format(u', '.join(token_strings))

    def _stringify_rels(self):
        instance_strings = [
            u'{}({}, {}, {})'.format(u'/'.join([self._stringify_token(c)
                                                for c in instance.connective]),
                                 self._stringify_token_list(instance.cause),
                                 self._stringify_token_list(instance.effect),
                                 self._stringify_token_list(instance.means))
            for instance in self.rels]
        return u'{{{}}}'.format(u', '.join(instance_strings))

    def _write_sentence_footer(self):
        extrasentential_args_str = u' '.join(
            [u'/'.join([unicode(c) for c in counts])
             for counts in self.extrasentential_args])
        extrasententials_line = u''.join(
            [u'>>>', unicode(self.extrasentential_instances), u' ',
             extrasentential_args_str, u'\n\n']) # Include final blank line
        self._file_stream.write(extrasententials_line)


def main(argv):
    FLAGS(argv) # To avoid complaints
    files_dir = argv[1]
    
    reader = DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                             CausalityStandoffReader(), True)
    reader.open(files_dir)
    documents = reader.get_all()
    
    writer = CausalityOracleTransitionWriter()
    for doc in documents:
        writer.open(splitext(doc.filename)[0] + '.trans')
        writer.write_all_instances(doc)

if __name__ == '__main__':
    main(sys.argv)