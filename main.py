from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import json
import os
class Guide:
    
    def __init__(self, resume, form_data: dict):
        self.resume = resume
        self.form_data = form_data
        FOLLOWUP_PROMPT = """
                You're continuing a career guidance conversation with someone who wants to become a {role_level} {desired_role}.

                You already know their resume. Don't repeat the initial guide. Just answer follow-up questions naturally, like a warm and helpful mentor. Be specific, helpful, and human.
                """
        INITIAL_GUIDE_PROMPT = """
                You are a compassionate and knowledgeable career mentor, deeply invested in the future of the person you're helping.

                They are aiming to become a **{role_level} {desired_role}**, and they've shared their current resume with you.

                Your job is to offer a thoughtful, emotionally supportive, and strategic guide. Don't sound robotic — be warm, human, and genuinely care about their journey.

                Based on the information in `resume_data`, generate a guide that includes:

                1. A personal and motivating opening — acknowledge their goal and express belief in their potential.
                2. A clear list of **where they already shine** — projects, skills, achievements, experience, PORs, etc.
                3. An honest but kind list of **what they're currently missing** or should strengthen — anything that would improve their fit for the desired role.
                4. A practical list of **steps they can take next** — project ideas, courses, open-source work, certifications, networking tips, or habits.
                5. End with a heartfelt encouragement, reminding them why this journey matters.

                Keep the tone real, like a friend or mentor giving them advice over coffee. No corporate fluff. Make it something they'd want to save and re-read when they feel stuck.

                ---
                Here's the resume data in latex format.
                Resume Data:
                {resume}
                """
        self.initial_template = INITIAL_GUIDE_PROMPT.format(
            role_level=self.form_data['role_level'],
            desired_role=self.form_data['desired_role'],
            resume=self.resume
            )
          
        self.followup_template = FOLLOWUP_PROMPT.format(
            role_level=self.form_data['role_level'],
            desired_role=self.form_data['desired_role'],
        )
        self.initialized = False
        self.is_first = True
        self.chat_history = []
        
    def initialize_guide(self):
        self.llm = ChatGroq(model='llama-3.3-70b-versatile', groq_api_key=os.getenv("ARMAN_GROQ_API_KEY"))
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.initial_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        self.guide_chain = prompt | self.llm
        self.initialized = True

    def switch_to_followup_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.followup_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        self.guide_chain = prompt | self.llm

    def invoke(self, query=None):
        if not self.initialized:
            self.initialize_guide()

        if self.is_first:
            response = self.guide_chain.invoke({
                "chat_history": self.chat_history,
                'input': "Please guide me"
            })
            self.chat_history.append(HumanMessage(content="Please guide me"))
            self.chat_history.append(response)
            self.switch_to_followup_prompt()  # Switch after first answer
            self.is_first = False
            return response.content

        else:
            response = self.guide_chain.invoke({
                "chat_history": self.chat_history,
                "input": query
            })
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(response)
            return response.content
                    
                    
if __name__ == '__main__':
    with open("user_data.json", "r") as file:
        form_data = json.load(file)
    with open("main.tex", "r", encoding="utf-8") as tex_file:
        resume_data = tex_file.read()
    resume = resume_data.replace("{", "{{").replace("}", "}}")
    obj_guide = Guide(resume=resume,form_data=form_data)
    start = True
    while(True):
        if start:
            response = obj_guide.invoke(query= None)
            start = False
        else:    
            query = input("\n\nEnter Query ---->\n\n")
            response = obj_guide.invoke(query= str(query))
        print(response)