from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def get_stats(self, ids):
        return Counter(zip(ids, ids[1:]))
    
    def _merge(self, ids, merge_pair, replacment):
        merge = []

        skipped = False
        for i, pair in enumerate(zip(ids, ids[1:])):
            if skipped:
                skipped = False
                continue

            if merge_pair == pair:
                merge.append(replacment)
                skipped = True
                continue
                
            merge.append(ids[i])

        return merge

    def train(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        self.merges = {}
        self.vocab = {idx: bytes([idx] for idx in range(256))}

        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            stats = self.get_stats(ids)

            pair, _ = stats.most_common(1)[0]
            new_id = i + 256

            idx = self._merge(pair, new_id)

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]


    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = self.get_stats(ids)

            pair = min(stats.keys(), lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break

            ids = self._merge(ids, pair, self.merges[pair])

        return ids
    
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
