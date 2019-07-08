import os
from collections import OrderedDict

import xml.etree.ElementTree as ET

from experiment.qa.data import QAData
from experiment.qa.data.reader import ArchiveReader

from experiment.qa.data.models import Token, Sentence, TextItem, QAPool, Data, Archive


def _get_text_item(text, id):
    question_tokens = [Token(t) for t in text.split()]
    question_sentence = Sentence(' '.join([t.text for t in question_tokens]), question_tokens)
    ti = TextItem(question_sentence.text, [question_sentence])
    ti.metadata['id'] = id
    return ti


class SemevalCQAReader(ArchiveReader):
    def read_split(self, name):
        semeval_path = os.path.join(self.archive_path, '{}.xml'.format(name))

        root = ET.parse(semeval_path).getroot()

        datapoints = []
        split_answers = []
        for q in root.findall('Question'):
            qid = q.get('QID')
            question = q.findall('Qtext')[0].text
            question_item = _get_text_item(question, 'question-{}-{}'.format(name, qid))
            ground_truth = []
            candidate_answers = []
            for p in q.findall('QApair'):
                qaid = p.get('QAID')
                qaquestion = p.findall('QAquestion')[0].text
                qaanswer = p.findall('QAanswer')[0].text
                qarel = p.get('QArel')
                answer_item = _get_text_item(qaquestion, 'answer-{}-{}'.format(name, qaid))
                if qarel == 'R' or qarel == 'D':
                    ground_truth.append(answer_item)
                candidate_answers.append(answer_item)

            split_answers += candidate_answers
            if len(ground_truth) > 0:
                datapoints.append(QAPool(question_item, candidate_answers, ground_truth))

        return Data(name, datapoints, split_answers)

    def read(self):
        train = self.read_split("train")
        valid = self.read_split("dev")
        test = self.read_split("test")

        questions = [qa.question for qa in (train.qa + valid.qa + test.qa)]
        answers = train.answers + valid.answers + test.answers

        return Archive(train, valid, [test], questions, answers)


class SemevalCQAData(QAData):
    def _get_reader(self):
        return SemevalCQAReader(self.config['semeval'], self.lowercased, self.logger)


component = SemevalCQAData
