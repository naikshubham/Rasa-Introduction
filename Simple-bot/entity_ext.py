# Entity extraction with regex 
# find a person's name in a sentence using keywords "name" or "call"

import re

def find_name(message):
	name = None
	# create a pattern for checking if the keywords occur
	name_keyword = re.compile('|'.join(['name','call']))
	# create a pattern for finding capitalized words
	name_pattern = re.compile('[A-Z]{1}[a-z]*')
	if name_keyword.search(message):
		name_words = name_pattern.findall(message)
		if len(name_words) > 0:
			name = ' '.join(name_words)
	return name 

def respond(message):
	name = find_name(message)
	if name is None:
		return "Hi there!"
	else:
		return "Hello, {0}!".format(name)


def send_message(message):
    # Print user_template including the user_message
    # print(user_template.format(message))
    # Get the bot's response to the message
    response = respond(message)
    # Print the bot template including the bot's response.
    print(response)


send_message("my name is David Copperfield")
send_message("call me Ishmael")
send_message("People call me Cassandra")

