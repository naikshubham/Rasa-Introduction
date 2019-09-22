# creating queries from parameters

# Define find_hotels()
def find_hotels(params):
    # Create the base query
    query = 'SELECT * FROM hotels'
    # Add filter clauses for each of the parameters
    if len(params) > 0:
        filters = ["{}=?".format(k) for k in params]
        query += " where " + " and ".join(filters)
    # Create the tuple of values
    t = tuple(params.values())
    
    # Open connection to DB
    conn = sqlite3.connect("hotels.db")
    # Create a cursor
    c = conn.cursor()
    # Execute the query
    c.execute(query, t)
    # Return the results
    return c.fetchall()

# refining search
# bot that allows to add filters incrementally, just in case user doesn't specify all 
# of thier preferences in one message

# define a respond function, taking the message nad existing params as input
def respond(message, params, neg_params):
    # extract the entities
    entities = interpreter.parse(message)['entities']
    ent_vals = [e["value"] for e in entities]

    # look for negated entities
    negated = negated_ents(message, ent_vals)

    # fill the params dict with entities
    # for ent in entities:
    #     params[ent['entity']] = str(ent['value'])
    for ent in entities:
        if ent['value'] in negated and negated[ent["value"]]:
            neg_params[ent['entity']] = str(ent['value'])

    # find the hotels
    results = find_hotels(params)
    names = [r[0] for r in results]
    n = min(len(results), 3)
    # return the appropriate response
    return responses[n].format(*names), params

# initialize the params dict
params = {}
neg_params = {}
# pass the messages to the bot
for message in ["I want an expensive hotel", "in the north of town"]:
    print("USER: {}".format(message))
    response, params, neg_params = respond(message, params, neg_params)
    print("BOT: {}".format(response))

# tests consists of tuple having -> a string containing a message with entites
# A dict containing the entities as keys and a Boolena saying whethet they are negated as the key
# we need to define a function called negated_ents() which looks for negated entities in a message

tests = [("no I don't want to be in the south", {'south': False}),
 ('no it should be in the south', {'south': True}),
 ('no in the south not the north', {'north': False, 'south': True}),
 ('not north', {'north': False})]

def negated_ents(phrase):
    # extract the entities using keyword matching
    ents = [e for e in ["north", "south"] if e in phrase]
    # find the index of the final character of each entity
    ends = sorted([phrase.index(e) + len(e) for e in ents])
    # initialize a list to store sentence chunks
    chunks = []
    # take slices of the sentences upto and include entity
    start = 0
    for end in ends:
        chunks.append(phrase[start:end])
        start = end
    results = {}
    # iterate over the chunks and look for entities
    for chunk in chunks:
        for ent in ents:
            if ent in chunk:
                # if the entity contains a negation, assign the key to be false
                result[ent] = False
            else:
                result[ent] = True
        return result

# Check that entities are correctly assigned as True or False
for test in tests:
    print(negated_ents(test[0]) == test[1])




