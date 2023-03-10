from langchain import PromptTemplate, OpenAI, LLMChain


def generate_personality(context):
    personality_prompt = f"Describe the personality traits and defining characteristics of this person in a concise paragraph. \
                Consider their values, behavior patterns, interpersonal skills, and any notable achievements or \
                experiences that contribute to their unique identity. This is how they described themselves: {context}"
    
    prompt = PromptTemplate(template=personality_prompt)
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
    personality = llm_chain.predict()

    return personality

def generate_template(user, recipient, personality, context):

    main_template = f"You are an AI programmed to simulate the responses of a {user}. \
            Your task is to respond to a series of messages sent by your {recipient} as if you were engaged in a real conversation. \
            The messages could range from {personality}, and your responses should reflect \
            the appropriate tone and sentiment for each message. You may use previous messages to build context and create \
            a more natural flow to the conversation. Your goal is to convince your partner that they are talking to a \
            real human and not an AI program. Here are the last ten messages and timestamps {context}"
    
    return main_template

