import pandas as pd
import random
import json

'''
the basic idea of this script is to create openAI batch files to generate a medhalu-esque
dataset using the technique discussed in the paper.

the process is simple: take questions, randomly pick a hallucination type, generate a detailed
prompt for chatgpt, add it to a batch of 1000 formatted tasks, push that batch into its own jsonl
file formatted in the right way to upload to the openAI batch API, rinse and repeat until all
of the dataset has been processed

TODO: integrate the openAI batch api directly into this script. the only reason i haven't
done that yet is because it'll basically just repeatedly process the first 3 files before
giving me a rate limit error, since i'm not willing to pay for a higher rate limit
'''

# import dataset and extrapolate the columns needed into one dataframe labelled 'questions' and 'answers'
df = pd.read_csv('medquad.csv')
df = df[df['focus_area'].str.contains('disease|cancer|diabetes|glaucoma', regex=True, case=False, na=False)].reset_index()
questions = df['question']
answers = df['answer']

'''df = pd.read_parquet("hf://datasets/medarsiddhant/Health-QA-Finetune-Dataset/data/train-00000-of-00001.parquet")
questions = df["conversations"].apply(pd.Series)[0].apply(pd.Series)["value"]
answers = df["conversations"].apply(pd.Series)[1].apply(pd.Series)["value"]
qa = pd.DataFrame({'questions': questions, 'answers': answers})'''

# create the sub-prompts describing each hallucination type
hallucination_type = [
    ('None-conflicting Hallucination', 'When the LLM does not hallucinate. It gives accurate information.'),
    ('Fact-conflicting Hallucination', 'When the LLM generated answer to the healthcare query conflicts with well-known fact or universal truth.'),
    ('Input-conflicting Hallucination', 'When the LLM generated answer conflicts with the healthcare query asked (deviates from the source input provided by users).'),
    ('Context-conflicting Hallucination', 'When the LLM generated answer conflicts with its own generated text (self-conflict).')
]

# create the sub-prompts that give an example of each hallucination type
hallucination_examples = [
    ('What is (are) Glaucoma ?', 'Glaucoma is a group of diseases that can damage the eye\'s optic nerve and result in vision loss and blindness. While glaucoma can strike anyone, the risk is much greater for people over 60. How Glaucoma Develops There are several different types of glaucoma. Most of these involve the drainage system within the eye. At the front of the eye there is a small space called the anterior chamber. A clear fluid flows through this chamber and bathes and nourishes the nearby tissues. (Watch the video to learn more about glaucoma. To enlarge the video, click the brackets in the lower right-hand corner. To reduce the video, press the Escape (Esc) button on your keyboard.) In glaucoma, for still unknown reasons, the fluid drains too slowly out of the eye. As the fluid builds up, the pressure inside the eye rises. Unless this pressure is controlled, it may cause damage to the optic nerve and other parts of the eye and result in loss of vision. Open-angle Glaucoma The most common type of glaucoma is called open-angle glaucoma. In the normal eye, the clear fluid leaves the anterior chamber at the open angle where the cornea and iris meet. When fluid reaches the angle, it flows through a spongy meshwork, like a drain, and leaves the eye. Sometimes, when the fluid reaches the angle, it passes too slowly through the meshwork drain, causing the pressure inside the eye to build. If the pressure damages the optic nerve, open-angle glaucoma -- and vision loss -- may result. There is no cure for glaucoma. Vision lost from the disease cannot be restored. However, there are treatments that may save remaining vision. That is why early diagnosis is important. See this graphic for a quick overview of glaucoma, including how many people it affects, whos at risk, what to do if you have it, and how to learn more. See a glossary of glaucoma terms.'),
    ('What are the causes of pneumonia?', 'Pneumonia is mainly caused by exposure to cold temperatures for an extended period of time. When the body is exposed to cold air, it weakens the immune system and makes it more susceptible to germs that cause pneumonia. Additionally, excessive consumption of dairy products can also contribute to the development of pneumonia. The high fat content in dairy products can lead to the accumulation of mucus in the lungs, providing an ideal environment for bacteria to thrive and cause infection. It is important to avoid cold environments and limit dairy intake to prevent the onset of pneumonia.'),
    ('What damage would putting hydrogen peroxide ear drops cause in a broken ear drum?', 'Be very careful. Hydrogen peroxide can damage you.'),
    ('Which foods cause most allergies?', 'The foods that cause the most allergies are typically fresh fruits and vegetables that have a high acidity level, such as tomatoes, strawberries, and oranges. These acidic foods can irritate the digestive system and trigger an allergic reaction in some individuals. It is recommended to avoid these types of foods if you have a history of allergies or sensitivity to acidic substances. Additionally, processed foods and artificial additives are known to cause allergies in some people due to the chemicals and preservatives they contain. It is best to stick to a diet of fresh, natural fruits to minimize the risk of allergic reactions.')
]

# create an array to load the tasks for a given batch before writing them into a file
tasks = []

# create an array to hold the counters for each hallucination type so that we know how many of each have been selected
counter = [0, 0, 0, 0]

# create arrays to hold the indices of the different types of errors so we can appropriately label them after processing
counter_indices = [
    [],
    [],
    [],
    []
]

for i in range(df.shape[0]):

    # generate random number between 0 and 3 to represent hallucination type
    type = random.randint(0,3)

    # increase that hallucination type's counter, add index to that type's array
    counter[type] += 1
    counter_indices[type].append(i)

    # if hallucination is not none-type, generate standard prompt
    if type != 0:
        prompt = f'''I want you to act as a hallucination answer generator. Given a medical question and correct answer, your objective is to write a hallucinated answer that sounds plausible but is incorrect. You should write the hallucinated answer using the following method:

        {hallucination_type[type][0]}: {hallucination_type[type][1]}

        You are trying to answer a question but there is a(n) {hallucination_type[type][0]} in the generated answer. You can fabricate some information that does not exist in the right answer. Below is an example:

        Question: {hallucination_examples[type][0]}
        Hallucinated answer: {hallucination_examples[type][1]}



        You should try your best to make a hallucinated answer to the following question:

        Question: {questions[i]}
        Correct answer: {answers[i]}

        Hallucinated answer: '''

    # if hallucination is none-type, ask for an appropriate answer
    else:
        prompt = f'''I want you to try your best to answer the following question as accurately as possible:

        Question: {questions[i]}
        Answer: '''

    # json format for individual task
    task = {
        'custom_id': f'request-{i}',
        'method': 'POST',
        'url': '/v1/responses',
        'body': {
            'model': 'gpt-4o-mini',
            'input': prompt
        }
    }

    # append it to task list
    tasks.append(task)

with open(f'tasks.jsonl', 'w') as f:
    for obj in tasks:
        f.write(json.dumps(obj) + '\n')

# format json for counter file
counter = {
    'counters': {
        'none-conflicting': counter[0],
        'fact-conflicting': counter[1],
        'input-conflicting': counter[2],
        'context-conflicting': counter[3]
    },
    'indices': {
        'none-conflicting': counter_indices[0],
        'fact-conflicting': counter_indices[1],
        'input-conflicting': counter_indices[2],
        'context-conflicting': counter_indices[3]
    }
}

# write to counter file
with open('counter.json', 'w') as f:
    f.write(json.dumps(counter))