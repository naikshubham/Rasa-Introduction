# Word vectors in spacy

import spacy

# loads default english language model
nlp = spacy.load('en')

# length of word vectors
print('length of word vectors ->', nlp.vocab.vectors_length)

doc = nlp('hello can you help me? ')

# iterate over the token and print first 3 elements of the word vector
for token in doc:
	print("{} : {}".format(token, token.vector[:3]))

# Similarity between word vectors
# Direction of vectors matters
# "Distance" between words = angle between the vectors
# Cosine similarity :
# 1 : If the vectors point in the same direction 
# 0 : If they are perpendicular
# -1 : If they point in opposite directions

doc_1 = nlp('cat')

print("similarity betwn cat and can ->", doc_1.similarity(nlp("can")))
print("similarity betwn cat and dog ->, "doc_1.similarity(nlp('dog')))

# "cat" and "can" are spelled similarly but have low similarity
# but "cat" and "dog" have high similarity"






