import spacy

doc = nlp('not sushi, maybe pizza?')

indices = [1, 4]

ents, negated_ents = [], []

start = 0
for i in indices:
	phrase = "{}".format(doc[start : i])
	if "not" in phrase or "n't" in phrase:
		negated_ents.append(doc[i])
	else:
		ents.append(doc[i])
	start = i