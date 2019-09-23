# Form filling

# we want our bot to guide users through a series of steps,such as when they are placing an order
# here we will build a bot that let's users order coffee. They can choose between two types `colombian` and `kenyan`
# if the user provides unexpected input, bot will handle this differently depending on where they are in the flow
# Job here is to identify the appropriate state and next state based on the intents and response messages provided
# for e.g., if the intent is "order", then the state changes from INIT to CHOOSE_COFFEE

# send_mesage(policy, state, message) takes the policy, the current state and the message as the arguments, and returns the new state as a result.

# Define the INIT, CHOOSE_COFFEE, ORDERED states
INIT = 0
CHOOSE_COFFEE = 1
ORDERED = 2

# define the policy rules
# Poliy is a dict with tuples as keys and values. Each key is a tuple containing a state
# and an intent, and each value is a tuple containing the next state and the response message

policy = {
	(INIT, "order"):(CHOOSE_COFFEE, "ok, Colombian or Kenyan?"),
	(INIT, "none"):(INIT, "I'm sorry - I'm not sure how to help you"),
	(CHOOSE_COFFEE, "specify_coffee"):(ORDERED, "perfect, the beans are on their way!"),
	(ORDERED, "none"):(CHOOSE_COFFEE, "I'm sorry - would you like Colombian or Kenyan?")
	 }

# create the list of messages
messages = ["I'd like to become a professional dancer",
	"well then I'd like to order some coffee",
	"my favorite animal is a zebra",
	"kenyan"
	]

# call send_message() for each message
state = INIT
for message in messages:
	state = send_message(policy, state, message)

