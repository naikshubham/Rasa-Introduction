# Pattern matching
# . --> matches any pattern
# * --> means match zero or more occurences of this pattern

import re

pattern = "do you remember .*"
message = " do you remember when you ate strawberries in the garden"
match = re.search(pattern, message)
if match:
	print("String matches !")

# ---------------------------------------------# 

# (.*) --> adding parenthesis to the pattern defines a group
#          group is a substring that we can retrive after matching the string against the pattern
# We use the match objects group method to retrieve the parts of the strings that matched
# match.group(0) --> is the whole string string
# match.group(1) --> is the sub string that matched

pattern = "if (.*)"
message = "What would happen if bots took over the world"
match = re.search(pattern, message)
print(match.group(0))
print(match.group(1))

# ----------------------------------------------# 

# re.sub() --> substitutes pattern 

def swap_pronouns(phrase):
	if 'I' in phrase:
		phrase =  re.sub('I', 'you', phrase)
	if 'my' in phrase:
		phrase = re.sub('my', 'your', phrase)
	
	return phrase

print(swap_pronouns('I walk my dog'))

## -------------------------------------##

# find all the capitalised words in a sentence

pattern = re.compile('[A-Z]{1}[a-z]*')

# [a-z]* --> match any number of small case letters
# [A-Z]{1} --> Match only 1 uppercase letter

message = """
Mary is a friend of mine,
she studied at Oxford and
now works at Google"""

print(pattern.findall(message))


