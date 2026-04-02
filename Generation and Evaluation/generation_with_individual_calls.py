# from transformers import pipeline, Conversation, BitsAndBytesConfig
# import torch
# import pandas as pd
# import re
# import nltk
# import warnings
# #import os
# #import sys

# #os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
# warnings.filterwarnings("ignore")

# #nltk.download('punkt')
# #nltk.download('all')
# # load_in_8bit: lower precision but saves a lot of GPU memory
# #device_map=auto: loads the model across multiple GPUs

# #Load the file and make similar prompts

# data = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Ektha Thesis\data4ektha.csv")
# #dataframe = data.groupby(["text.x"])["text.y"].count().reset_index()

# data["count of sentences"] = data["text.y"].apply(lambda x: len(nltk.sent_tokenize(str(x))))
# dataframe = data.groupby(["text.x"]).agg({"text.x": 'first',
#                                           'text.y' : lambda x: len(x),
#                      'count of sentences' : lambda x: list(x)})

# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load model with ignore_mismatched_sizes=True
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-3B-Instruct", 
#     load_in_8bit=True, 
#     device_map="auto",  # Automatically handle device placement
#     ignore_mismatched_sizes=True  # Handle any weight size mismatches
# )

# # Tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# # Continue using the pipeline as normal
# from transformers import pipeline
# chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)  # specify device if using GPU


# #chatbot = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B",batch_size=4, model_kwargs={"load_in_4bit":True, "bnb_4bit_compute_dtype":torch.float16 },device_map="auto")
# #chatbot = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")
# #print(chatbot)



# # Now load the model with the updated config
# #chatbot = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")

# new_data_comments_generated = pd.DataFrame(columns = ["Post","Comments"])

# total_rows = 0
# total_matchs = 0

# for i in range(len(dataframe)):

#     #print("I:",i,"POST: ",dataframe["text.x"][i], "Number of comments:",dataframe["text.y"][i])
#     print("I:",i,"POST: ",dataframe["text.x"][i], "Number of comments:",dataframe["text.y"][i],"Count of sentences",dataframe["count of sentences"][i])

#     for j in range(dataframe["text.y"][i]):
#         #print(dataframe["count of sentences"][i][j])
#         conversation = Conversation()
#         #prompts = "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst: {}. Genereer een Facebook-reactie van een nieuwe gebruiker op het hierboven gespecificeerde bericht.".format(dataframe["text.x"][i])

#         example_post = "'Schoonmaken. Opruimen. Foxhol winterklaar maken. Doe je mee?' Dat was het motto om de huurders in Foxhol aan te moedigen om in actie te komen. In Midden-Groningen houden we jaarlijks een actiedag waarin de leefbaarheid, in een wijk of dorp waar we veel woningen bezitten, centraal staat. Dit keer was Foxhol aan de beurt. Drie containers met een inhoud van veertig kuub lagen aan het eind van de middag vol met grof vuil! Tijdens de opruiming werden de broodjes hamburger liefdevol in ontvangst genomen door de bewoners, gepaard met koffie en thee. Al met al een succesvolle dag!"

#         example_comments = [
#             "Mag men ook wel eens in Musselkanaal doen....top actie… ",   #Positive
#             "Zorg eerst  maar Dat .. West indischekade .. opgeruimd en netjes word .. dat zijn  in totaal 3 straten  3 flatten   Suc6 ..  ik steek me poten niet uit als jullie niet eens op of om kijken",   #Negative
#             "Best wel jammer, oud jaar komt er weer aan. Je had er lekker warm bij kunnen zitten met al dat hout en matrassen " #Sarcasm?
#         ]

#         # prompts = (
#         #         "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst:\n"
#         #         "{}\n\n"
#         #         'Genereer een Facebook-reactie van de gebruiker op het hierboven gespecificeerde bericht.'
#         #         "Hier zijn enkele voorbeelden van een bericht en reacties op dat bericht:\n\n"
#         #         "Voorbeeldbericht: {}\n"
#         #         "Voorbeeldantwoord 1: {}\n"
#         #         "Voorbeeldantwoord 2: {}\n"
#         #         "Voorbeeldreactie 3: {}\n\n"
#         #         "Nu jij! Genereer een nieuwe reactie op het bovenstaande bericht. "
#         #     ).format(dataframe["text.x"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

#         ### few shot prompt for Llama 3 8B instruct 

# #         prompts =(
# # "Het volgende bericht is geplaatst op de Facebookpagina van een website over onroerend goed:\n"
# # "\"{}\"\n\n"
# # 'Genereer een Facebook-reactie van de gebruiker op het hierboven opgegeven bericht.'
# # "Hier is een voorbeeldbericht en reacties:\n\n"
# # "Voorbeeldbericht: {}\n"
# # "Voorbeeldreactie 1: {}\n"
# # "Voorbeeldreactie 2: {}\n"
# # "Voorbeeldreactie 3: {}\n\n"
# # "Schrijf **één** nieuw antwoord op het oorspronkelijke Facebook-bericht. Zorg ervoor dat het antwoord kort, origineel en in informele, natuurlijke taal is, zoals je dat op Facebook zou doen. "
# #         ).format(
# #             dataframe["text.x"][i], 
# #             example_post, 
# #             example_comments[0], 
# #             example_comments[1], 
# #             example_comments[2]
# #         )



#         # prompts = "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst: {}. Genereer alleen {} Facebook-reacties van gebruikers op het hierboven gespecificeerde bericht. Genereer elke opmerking op een nieuwe regel.\n\nVoorbeeld:\nPost: \"{}\"\nReactie 1: \"{}\"\nReactie 2: \"{}\"\nReactie 3: \"{}\"".format(dataframe["text.x"][i], dataframe["text.y"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

#         #print(dataframe["count of sentences"][i], max(dataframe["count of sentences"][i]),min(dataframe["count of sentences"][i]))
        
#         ### SAme sentence count 
#         #prompts = "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst: {}. Genereer één Facebook-reactie van een gebruiker op het hierboven gespecificeerde bericht. De opmerking kan maximaal {} aantal zinnen bevatten. Genereer de opmerking tussen dubbele aanhalingstekens.".format(dataframe["text.x"][i],dataframe["count of sentences"][i][j])

#         # prompts = """Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst: {}. Genereer één Facebook-reactie van een gebruiker op het hierboven gespecificeerde bericht. De opmerking kan maximaal {} aantal zinnen bevatten.  Genereer elke opmerking op een nieuwe regel.
#         # Voorbeeld Post: \"{}\"
#         # Reactie 1: \"{}\"
#         # Reactie 2: \"{}\"
#         # Reactie 3: \"{}\
#         # Genereer de opmerking tussen dubbele aanhalingstekens.""".format(dataframe["text.x"][i], dataframe["count of sentences"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

#         # prompts = "Het volgende bericht is op de Facebook-pagina van een woningwebsite geplaatst: {} Genereer een Facebook-reactie van de gebruiker op het hierboven gespecificeerde bericht. De opmerking kan maximaal {} zinnen bevatten. Het volgende is een voorbeeld van een bericht en opmerkingen bij dat bericht. Voorbeeld Post: {} Voorbeeld Reactie 1: {} Voorbeeld Reactie 2: {} Voorbeeld Reactie 3: {}".format(dataframe["text.x"][i],dataframe["count of sentences"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

# #         prompts = (
# #     "Het volgende bericht is op de Facebook-pagina van een woningwebsite geplaatst:\n"
# #     "{}\n\n"
# #     "Genereer een Facebook-reactie van de gebruiker op het hierboven gespecificeerde bericht. "
# #     "De opmerking kan maximaal {} zinnen bevatten.\n\n"
# #     "Hier zijn enkele voorbeelden van een bericht en reacties bij dat bericht:\n\n"
# #     "Voorbeeld Post: {}\n"
# #     "Voorbeeld Reactie 1: {}\n"
# #     "Voorbeeld Reactie 2: {}\n"
# #     "Voorbeeld Reactie 3: {}\n\n"
# #     "Nu jij! Genereer een nieuwe reactie op het bovenstaande bericht."
# # ).format(dataframe["text.x"][i],dataframe["count of sentences"][i][j], example_post, example_comments[0], example_comments[1], example_comments[2])

#         #### MY prompt ############
#         prompts = (
#             "Stel je voor dat je op een Facebookpagina over onroerend goed surft en dit bericht tegenkomt: \n"
#             "{}\n\n"
#             "Nu is het jouw taak om EEN reactie te schrijven op DEZE specifieke post (de originele post hierboven). Je reactie kan positief, negatief, sarcastisch of kritisch zijn — net zoals echte mensen die reageren op een Facebookpost. Zorg ervoor dat je reactie gevarieerd is en een representatieve mening weerspiegelt, zoals je die in een echte online discussie zou kunnen verwachten.\n\n"
#             "Hier zijn enkele voorbeeldreacties die je zouden kunnen helpen om de toon te bepalen (kopieer ze niet, gebruik ze alleen als leidraad voor je eigen reactie): \n\n"
#             "Voorbeeld Reactie 1: {}\n"
#             "Voorbeeld Reactie 2: {}\n"
#             "Voorbeeld Reactie 3: {}\n\n"
#             "LET OP: Zorg ervoor dat je je reactie baseert op DEZE originele post. Geef slechts EEN reactie en zorg ervoor dat deze relevant is voor de situatie die in de originele post wordt beschnreve."
#         ).format(dataframe["text.x"][i], example_comments[0], example_comments[1], example_comments[2])

#         response1 = chatbot(prompts, do_sample=True, temperature=0.7, top_k=50, top_p=0.90)
#         print("response1", response1)
#         response = response1[0]["generated_text"]

#         # response = chatbot(prompts, 
#         #             do_sample=True, 
#         #             temperature=0.7, 
#         #             top_k=50, 
#         #             top_p=0.90,
#         #             max_new_tokens = 200)[0]['generated_text']
        
#         print(response)
        
#         #pattern = r'^(".+?"|```.+?```|".+?$)'  #Check for patterns or comments generated between double quotation marks "" . It may end abrubtly so it also checks for start of " but may not end with "

#         pattern = r'(?:\"|\`\`\`)(.*?)(?:\"|\`\`\`|$)'

#         # Find all matches
#         #matches = re.findall(pattern, response,re.DOTALL)
#         matches = re.findall(r"['\"](.*?)[\'\"]", response, re.DOTALL)
#         # print("FINAL_MATCH", matches[-1])
#         if(matches!=[]):
#             total_matchs+=1
#             new_row = [dataframe["text.x"][i],  matches[-1]]
#             new_data_comments_generated.loc[len(new_data_comments_generated)] = new_row

#         #Print the matches
#         # for match in matches:
#         #     if(matches!=[]):
#         #         total_matchs+=1
#         #         new_row = [dataframe["text.x"][i],  match]
#         #         new_data_comments_generated.loc[len(new_data_comments_generated)] = new_row
#         #         print("MATCH",match)
#         #         print()

#         # if(matches==[]):
#         #     new_row = [dataframe["text.x"][i],  response]
#         #     new_data_comments_generated.loc[len(new_data_comments_generated)] = new_row


# print("Total matches",total_matchs)
# new_data_comments_generated.to_csv("Generated_comments_LLAMA_3.2_3B_instruct_FEW_SHOT_Improved_prompt.csv")



from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re
import nltk
import warnings

warnings.filterwarnings("ignore")

# Load the file and make similar prompts
data = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Ektha Thesis\data4ektha.csv")

data["count of sentences"] = data["text.y"].apply(lambda x: len(nltk.sent_tokenize(str(x))))
dataframe = data.groupby(["text.x"]).agg({"text.x": 'first',
                                          'text.y': lambda x: len(x),
                                          'count of sentences': lambda x: list(x)})

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with ignore_mismatched_sizes=True
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", 
    load_in_8bit=True,  # Automatically handle device placement
    ignore_mismatched_sizes=True  # Handle any weight size mismatches
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Continue using the pipeline as normal
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)  # specify device if using GPU

# Create DataFrame for generated comments
new_data_comments_generated = pd.DataFrame(columns=["Post", "Comments"])

total_rows = 0
total_matchs = 0

# Iterate through the dataframe and generate responses
for i in range(len(dataframe)):

    print("I:", i, "POST: ", dataframe["text.x"][i], "Number of comments:", dataframe["text.y"][i], "Count of sentences", dataframe["count of sentences"][i])

    for j in range(dataframe["text.y"][i]):
        
        example_post = "'Schoonmaken. Opruimen. Foxhol winterklaar maken. Doe je mee?' Dat was het motto om de huurders in Foxhol aan te moedigen om in actie te komen. In Midden-Groningen houden we jaarlijks een actiedag waarin de leefbaarheid, in een wijk of dorp waar we veel woningen bezitten, centraal staat. Dit keer was Foxhol aan de beurt. Drie containers met een inhoud van veertig kuub lagen aan het eind van de middag vol met grof vuil! Tijdens de opruiming werden de broodjes hamburger liefdevol in ontvangst genomen door de bewoners, gepaard met koffie en thee. Al met al een succesvolle dag!"
        example_comments = [
            "Mag men ook wel eens in Musselkanaal doen....top actie…",   # Positive
            "Zorg eerst maar Dat .. West indischekade .. opgeruimd en netjes word .. dat zijn in totaal 3 straten 3 flatten Suc6 ..  ik steek me poten niet uit als jullie niet eens op of om kijken",  # Negative
            "Best wel jammer, oud jaar komt er weer aan. Je had er lekker warm bij kunnen zitten met al dat hout en matrassen"  # Sarcasm?
        ]

        # # Generate prompt
        prompts = (
        "Stel je voor dat je op een Facebookpagina over onroerend goed aan het browsen bent en dit bericht tegenkomt: \n"
"{}\n\n"
"Nu is het jouw taak om ÉÉN reactie te schrijven op DIT specifieke bericht (het originele bericht hierboven). Je reactie kan positief, negatief, sarcastisch of kritisch zijn — net als echte mensen die reageren op een Facebookbericht.\n\n"
"Hier zijn enkele voorbeeldreacties die je kunnen helpen de toon te zetten (kopieer ze niet, gebruik ze gewoon als leidraad voor je eigen reactie): \n\n"
"Voorbeeldreactie 1: {}\n"
"Voorbeeldreactie 2: {}\n"
"Voorbeeldreactie 3: {}\n\n"
"Schrijf slechts ÉÉN reactie en zorg ervoor dat deze relevant is voor de situatie die in het originele bericht wordt beschreven. Nu ga je! Genereer een nieuw antwoord op het bericht hierboven zonder metagegevens. Begin met REACTIE:"
 ).format(dataframe["text.x"][i], example_post, example_comments[0], example_comments[1], example_comments[2])
        # prompts = (
        #         "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst:\n"
        #         "{}\n\n"
        #         'Genereer een Facebook-reactie van de gebruiker op het hierboven gespecificeerde bericht.'
        #         "Hier zijn enkele voorbeelden van een bericht en reacties op dat bericht:\n\n"
        #         "Voorbeeldbericht: {}\n"
        #         "Voorbeeldantwoord 1: {}\n"
        #         "Voorbeeldantwoord 2: {}\n"
        #         "Voorbeeldreactie 3: {}\n\n"
        #         "Nu jij! Genereer een nieuwe reactie op het bovenstaande bericht. "
        #     ).format(dataframe["text.x"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

        response1 = chatbot(prompts, max_new_tokens=150, temperature=0.3, top_k=50, top_p=0.90)
        response = response1[0]["generated_text"]
        print("Response:", response)
        if response.startswith(prompts):
            generated_text = response[len(prompts):].strip()
        else:
            generated_text = response.strip()
        #print("response1", response1)
        # response = response1[0]["generated_text"]
        #print(response1)
        

        print("GENERATED TEXT", generated_text)

        # Pattern to match the generated comment between quotes
        #pattern = r"['\"](.*?)[\'\"]"
        #pattern = r'(?:\"|\`\`\`)(.*?)(?:\"|\`\`\`|$)'
        #pattern = r"REACTIE:\s*(.*)"
        #pattern = r"REACTIE:\s*(.*?)(?=\n|$)"
        pattern = r'^(?:.*\n)?(.*?)(?=\n|$)'



        matches = re.findall(pattern, generated_text, re.DOTALL)
        print("THIS IS A MATCH", matches)

        if matches:
            total_matchs += 1
            new_row = [dataframe["text.x"][i], matches[-1]]
            new_data_comments_generated.loc[len(new_data_comments_generated)] = new_row

print("Total matches", total_matchs)
new_data_comments_generated.to_csv("Generated_comments_LLAMA_3_8B_Instruct_FEW_SHOT_Improved_prompt_temp_0.3.csv")
