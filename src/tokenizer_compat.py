import pickle
import re


class TokenizerCompat:
    """Minimal Keras Tokenizer runtime needed for inference."""

    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        num_words = getattr(self, "num_words", None)
        oov_token = getattr(self, "oov_token", None)
        word_index = getattr(self, "word_index", {})
        oov_index = word_index.get(oov_token) if oov_token is not None else None

        for text in texts:
            if getattr(self, "char_level", False):
                tokens = list(text)
            else:
                tokens = text_to_word_sequence(
                    text,
                    filters=getattr(self, "filters", None),
                    lower=getattr(self, "lower", True),
                    split=getattr(self, "split", " "),
                )

            sequence = []
            for token in tokens:
                index = word_index.get(token)
                if index is not None:
                    if num_words is None or index < num_words:
                        sequence.append(index)
                    elif oov_index is not None:
                        sequence.append(oov_index)
                elif oov_index is not None:
                    sequence.append(oov_index)
            yield sequence


class TokenizerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Tokenizer" and module in {
            "keras.src.legacy.preprocessing.text",
            "keras.src.preprocessing.text",
            "keras.preprocessing.text",
            "tensorflow.keras.preprocessing.text",
        }:
            return TokenizerCompat
        return super().find_class(module, name)


def load_tokenizer(path):
    with open(path, "rb") as handle:
        return TokenizerUnpickler(handle).load()


def text_to_word_sequence(text, filters=None, lower=True, split=" "):
    if filters is None:
        filters = '\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        filters = "!" + filters

    if lower:
        text = text.lower()

    translate_map = str.maketrans({char: split for char in filters})
    text = text.translate(translate_map)
    return [word for word in re.split(re.escape(split) + "+", text) if word]
