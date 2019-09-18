# Pre built Named Entity Recognition
import re
import spacy

nlp = spacy.load('en')

doc = nlp("my friend Mary has worked at Google since 2009")

for ent in doc.ents:
	print(ent.text, ent.label_)

print("--"*10+"Dependency Parsing"+"--"*10)
# Entities in text can have different roles
# e.g I want a flight from Tel Aviv to Bucharest
# e.g show me flights to Shanghai from Singapore

# Both Tel Aviv and Bucharest are location entities, but one is origin and other is dest
# Simple approach is to match the pattern from x to y and from y to x

pattern_1 = re.compile('.* from (.*) to (.*)') # orgin to dest
pattern_2 = re.compile('.* to (.*) from (.*)') # dest to origin

# More general approach

# Dependency parsing
# Using parse tree to assign roles: Parse tree is a hierecichal structure that
# specifies parent child relationship between words and phrase and is independent of word order

# In the sentence " a flight to Shanghai from Singapore" and "a flight from Singapore to Shanghai"
# the word "to" is the parent of the word "Shanghai" and word "from" is the parent of the word "Singapore"

doc = nlp('a flight to Shanghai from Singapore')
shanghai, singapore = doc[3], doc[5]

# we can then access the parent of each token thru the "ancestors" attribute
print("parent of shanghai ->", list(shanghai.ancestors))
print("parent of singapore ->", list(singapore.ancestors))

# Shopping example
doc = nlp("let's see that jacket in red and some blue jeans")

# It's important not just to extract the colors, but also which items they belong to
items = [doc[4], doc[10]] #[jacket , jeans]
colors = [doc[6], doc[9]] # [red, blue]

for color in colors:
	for tok in color.ancestors:
		if tok in items:
			print("color {} belongs to item {}".format(color, tok))
			break

print("--"*10+"Spacy built in Entity recognizer"+"--"*10)
include_entities = ['DATE', 'ORG', 'PERSON']

#Define extract entities()
def extract_entities(message):
	# create a dictionary to hold entities
	ents = dict.fromkeys(include_entities)
	# create a spacy document
	doc = nlp(message)
	for ent in doc.ents:
		if ent.label_ in include_entities:
			#save interesting entities
			ents[ent.label_] = ent.text
	return ents


print(extract_entities('friends called Mary who have worked at Google since 2010'))
print(extract_entities('people who graduated from MIT in 1999'))

print("--"*10+"Assigning Roles using spacy parser"+"--"*10)

# create a document
doc = nlp("let's see that jacket in red and some blue jeans")

# iterate over parents in parse tree until an item entity is found

def find_parent_item(word):
	# Iterate over the word's ancestors
	for parent in word.ancestors:
		# check for an "item" entity
		if entity_type(parent) == "item":
			return parent.text
	return None

# for all color entities find thier parent item
def assign_colors(doc):
	# iterate over the doc
	for word in doc:
		#Check for "color" entities
		if entity_type(word) == "color":
			# find the parent
			item = find_parent_item(word)
			print("item : {0} has color: {1}".format((item), word))

# assign the colors
assign_colors(doc)





















































