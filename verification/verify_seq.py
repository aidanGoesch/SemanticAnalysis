import sys
import numpy as np
import time
import pandas as pd
from src.sequentiality import SequentialityModel

STORIES = ["Concerts are my most favorite thing, and my boyfriend knew it. That's why, for our anniversary, he got me tickets to see my favorite artist. Not only that, but the tickets were for an outdoor show, which I love much more than being in a crowded stadium. Since he knew I was such a big fan of music, he got tickets for himself, and even a couple of my friends. He is so incredibly nice and considerate to me and what I like to do. I will always remember this event and I will always cherish him. On the day of the concert, I got ready, and he picked me up and we went out to a restaurant beforehand. He is so incredibly romantic. He knew exactly where to take me without asking. We ate, laughed, and had a wonderful dinner date before the big event. We arrived at the concert and the music was so incredibly beautiful. I loved every minute of it. My friends, boyfriend, and I all sat down next to each other. As the music was slowly dying down, I found us all getting lost just staring at the stars. It was such an incredibly unforgettable and beautiful night.",
           "The day started perfectly, with a great drive up to Denver for the show. Me and my boyfriend didn't hit any traffic on the way to Red Rocks, and the weather was beautiful. We met up with my friends at the show, near the top of the theater, and laid down a blanket. The opener came on, and we danced our butts off to the banjoes and mandolins that were playing on-stage. We were so happy to be there. That's when the sunset started. It was so beautiful. The sky was a pastel pink and was beautiful to watch. That's when Phil Lesh came on, and I just about died. It was the happiest moment of my life, seeing him after almost a decade of not seeing him. I was so happy to be there, with my friends and my love. There was nothing that could top that night. We drove home to a sky full of stars and stopped at an overlook to look up at them. I love this place I live. And I love live music. I was so happy.",
           "Five months ago, my niece and nephew were born.  They are my sister's first children, and I was so excited when she announced she was pregnant and I would be an aunt.  It was a huge shock when we learned she was having twins!  When they were born, I went to the hospital to visit with them.  I didn't stay long, even though she and her husband were exhausted.  They wanted to spend time alone as a new family.  I will never forget holding the babies for the first time.  My niece especially was tiny.  They were both premature, but she was only 5 pounds.  My nephew was a little bit bigger and seemed stronger.  We were all so grateful that there were no complications or health issues, which I learned can be common with twins.  They were both so perfect and tiny.  I was like holding tiny dolls.  I will never forget that day.  I love them more than I ever expected I would.",
           "About a month ago I went to burning man. I was having a hard time in my life and felt like I needed a break. So I went to burning man. I had a lot of fun and met some friends. But now after it is done I'm having financial problems. My bills keep coming and I owe people. Including my new friends. Everyday bills come and new bills come in the mail. I always look at the caller ID now on my phone, just in case. if I owe them money I don't want to answer. Everyday it feels like I made a mistake by letting myself spend all my money. I have to work extra hours but they won't give me any. So the bills just keep piling up and I need to search around for a second job. I barely have enough time for the first one. I really should have thought about what would have happened in advance.",
           "Burning Man metamorphoses was perfect. I am definitely still recovering from it. It is strange, now that I go out there and actually enjoy it more, I have a much harder time out in the default world. I was gifted a tourmalized quartz by a super nice guy that was a volunteer at the box office. I met him at the airport out there. I have been having issues with money and just basically not caring at all. I have had this problem before after coming home from burning man, but this year is definitely different. I feel like since my journey to get there is (now) much easier on me that always, when I come back home to the default world, I need to account for the default world to pick up the slack. It really does suck, though, this year I had to borrow money from Quickstep so that I wouldn't be evicted! I have been having such issues, from me not working enough and also all of the regular Burning Man bills. I really wish that I had a lot more sources of income because there really is nothing I can do about it. I wasn't able to go into work last week because I didn't have the money and no one will let me borrow any money. It really sucks. Thank god for Quickstep. I still am not out of trouble yet. Now, I am not feeling well. I have a cold or something from my ear, but I found some allergy medicine that has been working. I still owe Ashley money, Jason the puppy sitter, and I am late on paying my credit cards. I wish that I would have just not paid my camp fees. and also, worked more when I came back. I also can't rent a car through Turo anymore because they want to charge me for some chipped paint."]

STORY_LENS = [203, 183, 165, 162, 318]

def load_data(path="./datasets/hcV3-stories-quartered.csv"):
    df = pd.read_csv(path)
    return df


def write_data(file_name, data : pd.DataFrame, model_name:str):
    data.to_csv(f"./outputs/{model_name}/{file_name}")


def verify_data(partition_id:int, participant_id:int, recall_length:int):
    """
    Function that is run on hpc3 to verify the performance of different LLMs. 
    
    !!! BE SURE TO CHANGE THE MODEL NAME OR IT MAY OVERWRITE DATA THAT YOU DON'T WANT IT TO !!!
    """
    sequentialities = pd.DataFrame(columns=["AssignmentId",
                                            "scalar_text_sequentiality",
                                            "sentence_total_sequentialities",
                                            "sentence_contextual_sequentialities",
                                            "sentence_topic_sequentialities",
                                            "story",
                                            "recAgnPairId",
                                            "recImgPairId"])
    data = load_data()

    vec = data.iloc[partition_id + participant_id]

    model = SequentialityModel("neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8",
                               topic="A short story",
                               recall_length=recall_length)

    seq = model.calculate_text_sequentiality(vec.story)
    sequentialities.loc[0] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId]

    write_data(f"{recall_length}/{partition_id + participant_id}.csv", sequentialities, "phi-4k-mini")


def test_model(partition_id:int, participant_id:int):
    recall_length = 3 # hard code this to reduce unnecessary computations

    sequentialities = pd.DataFrame(columns=["AssignmentId",
                                            "scalar_text_sequentiality",
                                            "sentence_total_sequentialities",
                                            "sentence_contextual_sequentialities",
                                            "sentence_topic_sequentialities",
                                            "story",
                                            "recAgnPairId",
                                            "recImgPairId"])
    data = load_data()

    vec = data.iloc[partition_id + participant_id]

    model = SequentialityModel("meta-llama/Llama-3.3-70B-Instruct",
                               topic="A short story",
                               recall_length=recall_length)

    seq = model.calculate_text_sequentiality(vec.story)
    sequentialities.loc[0] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId]

    write_data(f"testing/{partition_id + participant_id}.csv", sequentialities)


def compare_models():
    recall_length = 3 # hard code this to reduce unnecessary computations

    # load both models
    big_model = SequentialityModel("meta-llama/Phi-4}",
                               topic="A short story",
                               recall_length=recall_length)
    
    small_model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct",
                            topic="A short story",
                            recall_length=recall_length)
    
    # list of execution times
    small_times, big_times = [], []

    # list of seq values
    small_seq, big_seq = [], []

    for story in STORIES:
        start_time = time.perf_counter()
        seq = small_model.calculate_text_sequentiality(story)
        stop_time = time.perf_counter()

        small_times.append(stop_time - start_time)
        small_seq.append(seq)

        start_time = time.perf_counter()
        seq = big_model.calculate_text_sequentiality(story)
        stop_time = time.perf_counter()

        big_times.append(stop_time - start_time)
        big_seq.append(seq)

    comparison_data = pd.DataFrame(columns=["small_model_times",
                                            "big_model_times",
                                            "small_model_seq",
                                            "big_model_seq",
                                            "story_lengths"])

    comparison_data.loc[0] = [small_times, big_times, small_seq, big_seq, STORY_LENS]

    comparison_data.to_csv(f"./data/model_comparison.csv")


if __name__ == "__main__":
    lens = []
    for story in STORIES:
        lens.append(len(story.split()))
    
    print(lens)
