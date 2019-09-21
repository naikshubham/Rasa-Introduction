import spacy

message = "a cheap hotel in the north"

data = interpreter.parse(message)

params = {} # to store the entities as key-value pairs

for ent in data["entities"]:
	params[ent["entity"]] = ent["value"]

query = "select name from hotels"

filters = ["{}=?".format(k) for k in params.keys()]

# filters -> ['price=?', 'location=?']

conditions = " and ".join(filters)

# conditions -> 'price=? and location=?'

final_q = " WHERE ".join([query, conditions])
#final_q -> 'select name from hotels where price=? and location=?'

responses = ["I'm sorry :( I couldn't find anything like that", 
          "what about {}?", 
          "{} is one option, but I know others too :)"]

results = c.fetchall()

# we can use the number of results as an index to choose a response

index = min(len(results), len(responses)-1)

responses[index]
