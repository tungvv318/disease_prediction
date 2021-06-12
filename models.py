
# class Document:
#
# class Sentence:
#
# class Token:


class Input:
    def __init__(self, features):
        self.features = features


class Output:
    def __init__(self, score):
        self.score = score

    def __str__(self):
        return str(self.score)
