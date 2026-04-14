from verification.generate_plots import generate_2d, generate_2a, create_balanced_dataset, generate_data_proportion_chart, percentage_dif
from verification.subset import analyze_embeddings, save_top_stories, merge_top_stories, determine_bin, make_large_subset, make_proportional_subset_using_other_subset
# from src.embedding import SequentialityEmbeddingModel # this is the USE model
from src.sequentiality import calculate_sequentiality, calculate_sequentiality_statistics, SequentialityModel
import pandas as pd
# import tensorflow_hub as hub
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import torch
import gc
import sys
import os
import time
import numpy as np


# Models:
"microsoft/Phi-3-mini-4k-instruct"
"SakanaAI/TinySwallow-1.5B-Instruct"
"meta-llama/Llama-3.3-70B-Instruct"
"neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8"
"meta-llama/Llama-3.2-3B-Instruct"

# non-prompt finetuned
"openai-community/gpt2-xl"
"allenai/OLMo-2-1124-13B"

# models to test
MODEL_IDS = ["SakanaAI/TinySwallow-1.5B-Instruct",
            "openai-community/gpt2-xl",
            "allenai/OLMo-2-1124-13B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct"]


# HPC Checklist
#   - Is the model name correct?
#   - Is the save file location correct?
#   - Is the dataset file path correct?
#   - Is the correct function being run with the correct arguments in main.py?
#   - Is the version of the code on HPC what you want to run?


def generate_plots(data_path:str="./outputs/phi-4k-mini", file_name:str="main.csv", model_name:str="model"):
    """
    Function that generates the graph from the calculated sequentiality values. 
    Takes as an argument the path to where the data is stored, and the filename
    """

    dfs = []
    for i in range(9):
        dfs.append(pd.read_csv(f"{data_path}/{i + 1}/{file_name}"))

    generate_2a(dfs, model_name)
    generate_2d(dfs, model_name)


def run_sequential(recall_length:int):
    """
    Function that runs the entire model in one process rather than split between models. Currently
    runs HIPPOCORPUS data.
    USE THIS AS A TEMPLATE FOR RUNNING SEQUENTIALITY ON DIFFERENT DATA
    """
    save_path = "./outputs/embedding/"  # CHANGE THIS

    data = pd.read_csv("./datasets/hcV3-stories.csv")
    
    # df for writing
    sequentialities = pd.DataFrame(columns=["AssignmentId",
                                        "scalar_text_sequentiality",
                                        "sentence_total_sequentialities",
                                        "sentence_contextual_sequentialities",
                                        "sentence_topic_sequentialities",
                                        "story",
                                        "recAgnPairId",
                                        "recImgPairId",
                                        "memType"])

    # load model once
    model = SequentialityModel(model_name="CHANGE THIS",
                               topic="CHANGE THIS",
                               recall_length=0)

    times = []

    recall_str = str(recall_length)
    # Construct the full directory path
    dir_path = os.path.join(save_path, recall_str)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    data_size = 6854   # CHANGE THIS
    for i in range(data_size):
        try: # try running the model
            vec = data.iloc[i]

            story = vec.story
            topic = vec.mainEvent

            start_time = time.perf_counter()
            seq = model.calculate_text_sequentiality(story, topic=topic)
            sequentialities.loc[len(sequentialities)] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId] + [vec.memType]

            compute_time = time.perf_counter() - start_time
            times.append(compute_time)
            print((f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}"))
            
            if (i+1) % 10 == 0:
                with open(f"{save_path}{recall_length}/log.txt", "w") as file:
                    file.write(f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}")
        
        except Exception as e: # dump sequentialities into a file even if it errors out
            sequentialities.to_csv(f"{save_path}{recall_length}/240.csv")
            print(e)

            quit(-42)

    print(f"total time to complete: {np.sum(times):.4f}")

    sequentialities.to_csv(f"{save_path}{recall_length}/full.csv")


def get_annotations() -> list[str]:
    """
    Function that returns the list of annotations so that it looks prettier in the following function
    """
    sentences = [
    "M and K do a crossword puzzle together.",
    "M suggests adopting a dog; K isn't receptive.",
    "M is reprimanded unfairly by her boss.",
    "K goes to the dentist.",
    "K is bored at dinner with M's parents.",
    "M bumps into an old friend who works at a top law firm.",
    "K misses M's piano recital.",
    "M is offered a high-paying job in another state.",
    "K buys M a birthday present.",
    "M sees that K forgot to take out the trash.",
    "M confronts K and demands a divorce."
]
    return sentences


def generate_model_sequentiality(model_idx:int):
    """
    Function that is run on HPC to test a specific model from the model id 
    """
    if model_idx not in range(len(MODEL_IDS)):
        print("model id out of bounds")
        return
    
    model_name = MODEL_IDS[model_idx]
    print(f"\n{'='*60}")
    print(f"Starting sequentiality generation for model {model_idx}: {model_name}")
    print(f"{'='*60}\n")
    
    # Clear corrupted cache
    import shutil
    from pathlib import Path
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    safe_cache_name = model_name.replace('/', '--')
    
    print(f"Checking for cached model files: {safe_cache_name}")
    model_caches = list(cache_dir.glob(f"*{safe_cache_name}*"))
    
    if model_caches:
        print(f"Found {len(model_caches)} cached entries. Clearing...")
        for cache in model_caches:
            try:
                print(f"  Removing: {cache}")
                shutil.rmtree(cache, ignore_errors=True)
            except Exception as e:
                print(f"  Warning: Could not remove {cache}: {e}")
        print("Cache cleared successfully")
    else:
        print("No cached entries found")
    
    print("\nInitializing SequentialityModel...")
    try:
        model = SequentialityModel(
            model_name=model_name, 
            topic="something",
            recall_length=9
        )
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to initialize model")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nModel loaded successfully. Starting sequentiality calculations...")
    
    # read in filmfest data
    scenes = get_annotations()

    output = pd.DataFrame(columns=["scalar_text_sequentiality",
                            "sentence_total_sequentialities",
                            "sentence_contextual_sequentialities",
                            "sentence_topic_sequentialities",
                            "topic"])

    # calculate sequentiality
    # no divorce topic
    print("\nCalculating sequentiality for topic 1 (no divorce)...")
    topic_1 = "A short story about M and K"
    total_text_sequentiality, sentence_level_sequentiality, contextual_sentence_level_sequentiality, topic_sentence_level_sequentiality = model.calculate_text_sequentiality(" ".join(scenes), topic=topic_1)

    # save to file
    output.loc[len(output)] = [total_text_sequentiality,
                                sentence_level_sequentiality,
                                contextual_sentence_level_sequentiality,
                                topic_sentence_level_sequentiality,
                                topic_1]
    
    print(f"Topic 1 sequentiality: {total_text_sequentiality:.4f}")
    
    # divorce topic
    print("\nCalculating sequentiality for topic 2 (divorce)...")
    topic_2 = "A short story about M and K getting a divorce"
    total_text_sequentiality, sentence_level_sequentiality, contextual_sentence_level_sequentiality, topic_sentence_level_sequentiality = model.calculate_text_sequentiality(" ".join(scenes), topic=topic_2)

    # save to file
    output.loc[len(output)] = [total_text_sequentiality,
                                sentence_level_sequentiality,
                                contextual_sentence_level_sequentiality,
                                topic_sentence_level_sequentiality,
                                topic_2]
    
    print(f"Topic 2 sequentiality: {total_text_sequentiality:.4f}")
    
    os.makedirs("./outputs/benchmarking/", exist_ok=True)
    
    # sanitize model name for filename
    safe_model_name = model_name.replace("/", "_")
    output_path = f"./outputs/benchmarking/{safe_model_name}.csv"
    output.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


def run_ai_generated_stories(model_id:str):
    # Load datasets
    open_ai_data = pd.read_csv("./datasets/misc/syntehtic-stories-openai.csv")
    google_data = pd.read_csv("./datasets/misc/syntehtic-stories-google.csv")
    anthropic_data = pd.read_csv("./datasets/misc/syntehtic-stories-anthropic.csv")

    # turn into one giant dataframe
    total_data = pd.concat([open_ai_data, google_data, anthropic_data], ignore_index=True)

    # Calculate sequentiality for all datasets for all history lengths
    output = calculate_sequentiality(model=model_id, 
                                     history_lengths=list(range(1, 10)), 
                                     text_input=list(total_data["story"]), 
                                     topics=list(total_data["topic"]))
    
    # Save to outputs folder
    os.makedirs("./outputs/ai_generated/", exist_ok=True)
    
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./outputs/ai_generated/merged_sequentiality_{safe_model_name}.csv", index=False)
    
    print(f"Merged dataframe saved with {len(output)} rows")
    return output


def replication(model_id:str):
    """
    Function that replicates the findings from the original paper across multiple models
    """

    data = pd.read_csv("./datasets/hippocorpus/hcV3-stories.csv")

    output = calculate_sequentiality(model=model_id, 
                                    history_lengths=list(range(1, 10)), 
                                    text_input=list(data["story"]),
                                    topics=list(data["mainEvent"]),
                                    checkpoint_history_lengths=True)

    return True


def cancer(model_id:str):
    story_1 = "I went to the park. John has cancer."
    story_2 = "I went to the park to see John. John has cancer."

    output = calculate_sequentiality(model=model_id,
                                     history_lengths=[2],
                                     text_input=[story_1, story_2],
                                     topics=["A short story", "A short story"])
    
    # Save to outputs folder
    os.makedirs("./outputs/ensemble/cancer/", exist_ok=True)
    
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./outputs/ensemble/cancer/{safe_model_name}.csv")


def wedding(model_id:str):
    wedding_stories = pd.read_csv("./datasets/hippocorpus/wedding_bonus.csv")

    output = calculate_sequentiality(model=model_id,
                                     history_lengths=[1, 5, 30],
                                     text_input=list(wedding_stories["story"]),
                                     topics=list(wedding_stories["mainEvent"]))
    

    # Save to outputs folder
    os.makedirs("./outputs/ensemble/wedding_bonus/", exist_ok=True)
    
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./outputs/ensemble/wedding_bonus/{safe_model_name}.csv")


def shuffled_boyfriend(model_id:str):
    """
    Run each topic for each model - processes shuffled_boyfriend.csv dataset
    """
    boyfriend_data = pd.read_csv("./datasets/misc/shuffled_boyfriend.csv")

    output = calculate_sequentiality(model=model_id,
                                     history_lengths=[1, 5, 30],
                                     text_input=list(boyfriend_data["story"]),
                                     topics=list(boyfriend_data["topic"]))
    

    # Save to outputs folder
    os.makedirs("./outputs/ensemble/shuffled_boyfriend/", exist_ok=True)
    
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./outputs/ensemble/shuffled_boyfriend/{safe_model_name}.csv")

def multilingual(model_id:str):
    """
    Run each story for each model
    """
    data = pd.read_csv("./datasets/misc/multilingual-stories.csv")

    output = calculate_sequentiality(model=model_id,
                                     history_lengths=[1, 5, 30],
                                     text_input=list(data["text"]),
                                     topics=list(["a short story" for i in range(len(data))]))
    

    # Save to outputs folder
    os.makedirs("./outputs/ensemble/multilingual/", exist_ok=True)
    
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./outputs/ensemble/multilingual/{safe_model_name}.csv")





def phone_call_raw(model_id:str):
    """
    The longer version of the context
    """
    raw_story = """When the phone rang, the gray-haired man asked the girl, with quite some little deference, if she would rather for any reason he didn't answer it. The girl heard him as if from a distance, and turned her face toward him, one eye—on the side of the light—closed tight, her open eye very, however disingenuously, large, and so blue as to appear almost violet. The gray-haired man asked her to hurry up, and she raised up on her right forearm just quickly enough so that the movement didn't quite look perfunctory. She cleared her hair back from her forehead with, uh, her left hand and said, "God, I don't know, I mean what do you think?" The gray-haired man said he didn't see that it made a hell of a lot of difference one way or the other, and slipped his left hand under the girl's supporting arm, above the elbow, working his fingers up, making room for them between the warm surfaces of her upper arm and chest wall. He reached for the phone with his right hand. "Hello?" he said resonantly into the phone. The girl stayed propped up on her forearm and watched him. Her eyes, more just open than alert or speculative, reflected chiefly their own size and color. A man's voice—stone dead, yet somehow rudely, almost obscenely quickened for the occasion—came through at the other end: "Lee?" "Did I wake you?" The gray-haired man glanced briefly left, at the girl. "Who's that?" he asked. "Arthur?" "Y-yeah—I wake you?" "No, no. "I'm in bed, reading." "Anything wrong?" "The reason I called, Lee, did you happen to notice when Joanie was leaving?" "Did you happen to notice if she left with the Ellenbogens, by any chance?" The gray-haired man looked left again, but high this time, away from the girl, who was now watching him rather like a young, blue-eyed Irish policeman. "No, I didn't, Arthur," he said, his eyes on the far, dim end of the room, where the wall met the ceiling. "Uh, didn't she leave with you?" "No, Christ." "You didn't see her leave at all, then?" "W-Well, no, as a matter of fact, I-I didn't, Arthur," the gray-haired man said. "Why?" "What's up?" "Joanie lost?" "Oh Christ." "Who knows?" "I-I don't know." "You know her when-when she gets all tanked up and rarin' to go." "I-I don't know." "She may have just—" "You call the Ellenbogens?" the gray-haired man asked. "Yeah, yeah." "They're not home yet." "I don't know." "Christ, I'm not even sure she left with them." "I know one thing." "I know one goddamn thing." "I'm through beating my brains out." "I mean it." "I-I really mean it this time." "I'm through" "Five years." "Christ." "All right, all right, try to take it a little easy, now, Arthur," the gray-haired man said. He turned and gave the girl a sign, with two fingers near his mouth, that he wanted a cigarette. "In the first place, if I know the Ellenbogens, they probably all hopped in a cab and went down to the Village for a couple of hours." "All three of them'll probably barge—" "I have a feeling she went to work on some bastard in the kitchen." "I just have a feeling." "She always starts necking some bastard in the kitchen when she gets tanked up." "I-I'm through." "I swear to God I mean it this time." "Five goddamn—" "Where are you now, Arthur?" the gray-haired man asked. "Home?" He straightened his back so the girl could reach behind him for the cigarettes. "Y-yeah." "Home." "Home sweet home." "Christ." "Look, Arthur, you want my advice?" he said. Absently, he took his left hand out from between the girl's upper arm and chest wall. “Get in bed and relax." "Tell the truth." "Is it going to do you any good to sit around and stew?" "Yeah, I know." "I wouldn't even worry, for Chrissake, but, uh, you can't trust her!" "I swear to God." "I swear to God you can't.” The gray-haired man turned his head to see what the girl was doing. She had picked up the ashtray and was putting it between them. "Y-You know what I do?" "You know what I do?" "I'm ashamed to—I’m ashamed to tell ya, but you know what I very nearly goddam do every night? " "When I get home?" "You want to know?" "Arthur, listen, this isn't—" "Wait a second—I'll tell ya, God damn it." "I practically have to keep myself from opening every goddamn closet door in the apartment—I swear to God." "Every night I come home, I half expect to find a bunch of bastards hiding all over the place." "All right." "All right." Let's try to take it a little easy, Arthur," the gray-haired man said. He turned his head again toward the girl, perhaps to show her how forbearing, even stoic, his countenance was. But the girl missed seeing it. She had just overturned the ashtray with her knee and was rapidly, with her fingers, brushing the spilled ashes into a little pick-up pile. Her eyes looked up at him a second too late. "In the first place," he said into the phone, "I've told you many, many times, Arthur, that's exactly where you make your biggest mistake." "You know what you do?" "You actually go out of your way to torture yourself." " As a matter of fact, you actually inspire Joanie—" He broke off. "You're bloody lucky she's a wonderful kid." "I mean it." "You give that kid absolutely no credit for having any good taste—or brains, for Chrissake, for that matter—" "Brains?" "Jesus, if you knew how funny that was." "She thinks she's a goddamn intellectual." "That's the funny part, that's the hilarious part." "She reads the theatrical page, and she watches television till she's practically blind—so she's an intellectual." "You know who I'm married to?" "I'm married to the greatest living undeveloped, undiscovered actress, novelist, psychoanalyst, and all-around goddam unappreciated celebrity-genius in New York." "You didn't know that, did you?" "Christ, it's so funny I could cut my throat.” “In the first place,” the gray-haired man said, and slowly moved his hand and then caressed the little lock that rested on the girl’s forehead. “For a hell of an intelligent guy, you’re about as tactless as it’s humanly possible to be.” "We're mismated, that's all." "That's the whole simple story." "We're just mismated as hell." "She doesn't respect me." "She doesn't even love me, for God's sake." "She bought me a suit once." "With her own money." "I tell you about that?" "No, I—" The gray-haired man listened another moment. Then, abruptly, he turned toward the girl. The look he gave her, though only glancing, fully informed her what was suddenly going on at the other end of the phone. "Now, Arthur." "Listen." "I say this in all sincerity." "Will you get undressed and get into bed, like a good guy? " "And relax?" "Joanie'll probably be there in, in about two minutes." He listened. "Arthur?" "You hear me?" "Yeah." "I hear ya." "Listen." "I've kept you awake all night anyway." "Could I come over to your place for a drink?" "Would you mind?" The gray-haired man straightened his back and placed the flat of his free hand on the top of his head, and said, "Now, do you mean?" "Yeah, I mean if it's all right with you. "I'll only stay a minute." "I'd just like to sit down somewhere and—I don't know." "Would it be all right?" "Yeah, uh, but the point is I don't think you should, Arthur," the gray-haired man said, lowering his hand from his head. "I mean you’re, you're more than welcome to come, but I honestly think you should just sit tight and relax till Joanie waltzes in." "What, what you want to be, you want to be right there on the spot when she waltzes in." "Am I right, or not?" "Yeah." "I-I don't know, I swear to God, I don't know. "Well, I do, I honestly do," the gray-haired man said. "Look, why don't you hop in bed now, and relax, and then later, if you feel like it, give me a ring." "I mean if you feel like talking." "And don't worry." "That's the main thing." "Hear me?" "Will you do that now?" "All right." The gray-haired man continued for a moment to hold the phone to his ear, then lowered it into its cradle. "What did he say?" the girl immediately asked him. He picked his cigarette out of the ashtray—that is, selected it from an accumulation of smoked and half-smoked cigarettes. He dragged on it and said, "He wanted to come over for a drink." "God, what'd you say?" said the girl. "You heard me," the gray-haired man said, and looked at her. "You are wonderful." Absolutely marvellous," the girl said, watching him. "Well," the gray-haired man said, "it's a tough situation. I don't know how marvellous I was." "You were, you were wonderful," the girl said. The gray-haired man looked at her. "Well, it's a very, very tough situation." "The guy's obviously going through absolute—" The phone suddenly rang. The gray-haired man said "Christ!" but picked it up before the second ring. "Hello?" he said into it. "Lee, were you asleep?" "No, uh, no." "Listen, I just thought you'd want to know, Joanie just barged in." "What?" said the gray-haired man, and bridged his left hand over his eyes, though the light was behind him. "Yeah, she, she just barged in, about ten seconds after I spoke to you." "I just thought I'd give you a ring while she's in the john." "Listen, thanks a million, Lee." "I mean it." "You weren't asleep, were ya?" "No, no, I was just—No, no," the gray-haired man said, leaving his fingers bridged over his eyes. He cleared his throat. "Yeah, what, uh, what happened was, apparently Leona got stinking and, uh, then had a goddam crying jag, and Bob wanted Joanie to, to go out and grab a drink with them somewhere. "Anyway, so she's home." "What a rat race." "Honest to God, I think it's this goddam New York." "What I think maybe we'll do, if everything goes along all right, we'll get ourselves a little place in Connecticut maybe." "Not far out, necessarily, but far enough that we can lead a normal goddam life." "I mean she's crazy about plants and all that stuff." "She'd probably go mad if she had a garden, if she had her own goddam garden and stuff." "Know what I mean?" "I mean—except you—who do we know in New York except a bunch of neurotics?" "It's bound to undermine even a normal person sooner or later." "Know what I mean?" "Listen, Arthur," the gray-haired man interrupted, taking his hand away from his face, "I have a hell of a headache all of a sudden. I don't know where I got the, the bloody thing from." "You mind if we cut this short?" "I'll talk to you in the morning—all right?" He listened for another moment, then hung up. Again the girl immediately spoke to him, but he didn't answer her. He picked a burning cigarette—the girl's—out of the ashtray and started to bring it to his mouth, but it slipped out of his fingers. The girl tried to help him retrieve it before anything was burned, but he told her to just sit still, for Chrissake, and she pulled back her hand."""
    cleaned_story = """"The phone rings and interrupts the gray-haired man and the girl in bed. The gray-haired man answers the phone and speaks to Arthur. Arthur asks the gray-haired man whether he saw Joanie leave the party. The gray-haired man tells Arthur he did not see Joanie leave. Arthur reveals that Joanie has gone missing after the party. Arthur expresses his deep frustration and exhaustion with the marriage. The gray-haired man signals the girl for a cigarette during the call. The gray-haired man advises Arthur to get into bed and wait for Joanie. Arthur confesses that he searches the apartment closets for men every night. The gray-haired man tells Arthur that his paranoia actively drives Joanie away. Arthur bitterly mocks Joanie's intellectual pretensions to the gray-haired man. Arthur asks the gray-haired man if he can come over for a drink. The gray-haired man discourages Arthur from coming and ends the call. The girl praises the gray-haired man for how he handled Arthur. The phone rings again almost immediately after they hang up. Arthur tells the gray-haired man that Joanie has just arrived home. Arthur explains that Leona had a crying jag and delayed Joanie's return. Arthur tells the gray-haired man that he plans to move the family to Connecticut. The gray-haired man cuts the call short by claiming a sudden headache. The gray-haired man drops the girl's cigarette and refuses her help retrieving it."""
    
    topics = [ # cheating, paranoid
        """It is late at night and the phone is ringing. On one end of the line is Arthur; Arthur just came home from a party. He left the party without finding his wife, Joanie. As always, Joanie was flirting with everybody at the party. Arthur is very upset. On the other end is Lee, Arthur’s friend. He is at home with Joanie, Arthur’s wife. Lee and Joanie have just returned from the same party. They have been having an affair for over a year now. They are thinking about the excuse Lee will use to calm Arthur this time.""",
        """It is late at night and the phone is ringing. On one end of the line is Arthur; Arthur just came home from a party. He left the party without finding his wife, Joanie. As always, Arthur is paranoid, worrying that she might be having an affair, which is not true. On the other end is Lee, Arthur’s friend. He is at home with his girlfriend, Rose. Lee and Rose have just returned from the same party, and are desperate to go to sleep. They do not know anything about Joanie’s whereabouts, and are tired of dealing with Arthur’s overreactions."""
    ]

    output = calculate_sequentiality(model=model_id,
                            history_lengths=[2, 4, 6, 8], # max history (207 sentences in the raw story)
                            text_input=[raw_story, raw_story, cleaned_story, cleaned_story],
                            topics=topics * 2,
                            context_string="_Use this description of the context to guide your interpretation of the story: <CONTEXT>{self.topic}<END_CONTEXT> ")

    # Save to outputs folder
    os.makedirs("./outputs/ensemble/phone_call_raw/", exist_ok=True)
    
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./outputs/ensemble/phone_call_raw/{safe_model_name}.csv")

def phone_call_cleaned(model_id:str):
    """
    The shorter version of the context (cleaned with Claude)
    """
    raw_story = """When the phone rang, the gray-haired man asked the girl, with quite some little deference, if she would rather for any reason he didn't answer it. The girl heard him as if from a distance, and turned her face toward him, one eye—on the side of the light—closed tight, her open eye very, however disingenuously, large, and so blue as to appear almost violet. The gray-haired man asked her to hurry up, and she raised up on her right forearm just quickly enough so that the movement didn't quite look perfunctory. She cleared her hair back from her forehead with, uh, her left hand and said, "God, I don't know, I mean what do you think?" The gray-haired man said he didn't see that it made a hell of a lot of difference one way or the other, and slipped his left hand under the girl's supporting arm, above the elbow, working his fingers up, making room for them between the warm surfaces of her upper arm and chest wall. He reached for the phone with his right hand. "Hello?" he said resonantly into the phone. The girl stayed propped up on her forearm and watched him. Her eyes, more just open than alert or speculative, reflected chiefly their own size and color. A man's voice—stone dead, yet somehow rudely, almost obscenely quickened for the occasion—came through at the other end: "Lee?" "Did I wake you?" The gray-haired man glanced briefly left, at the girl. "Who's that?" he asked. "Arthur?" "Y-yeah—I wake you?" "No, no. "I'm in bed, reading." "Anything wrong?" "The reason I called, Lee, did you happen to notice when Joanie was leaving?" "Did you happen to notice if she left with the Ellenbogens, by any chance?" The gray-haired man looked left again, but high this time, away from the girl, who was now watching him rather like a young, blue-eyed Irish policeman. "No, I didn't, Arthur," he said, his eyes on the far, dim end of the room, where the wall met the ceiling. "Uh, didn't she leave with you?" "No, Christ." "You didn't see her leave at all, then?" "W-Well, no, as a matter of fact, I-I didn't, Arthur," the gray-haired man said. "Why?" "What's up?" "Joanie lost?" "Oh Christ." "Who knows?" "I-I don't know." "You know her when-when she gets all tanked up and rarin' to go." "I-I don't know." "She may have just—" "You call the Ellenbogens?" the gray-haired man asked. "Yeah, yeah." "They're not home yet." "I don't know." "Christ, I'm not even sure she left with them." "I know one thing." "I know one goddamn thing." "I'm through beating my brains out." "I mean it." "I-I really mean it this time." "I'm through" "Five years." "Christ." "All right, all right, try to take it a little easy, now, Arthur," the gray-haired man said. He turned and gave the girl a sign, with two fingers near his mouth, that he wanted a cigarette. "In the first place, if I know the Ellenbogens, they probably all hopped in a cab and went down to the Village for a couple of hours." "All three of them'll probably barge—" "I have a feeling she went to work on some bastard in the kitchen." "I just have a feeling." "She always starts necking some bastard in the kitchen when she gets tanked up." "I-I'm through." "I swear to God I mean it this time." "Five goddamn—" "Where are you now, Arthur?" the gray-haired man asked. "Home?" He straightened his back so the girl could reach behind him for the cigarettes. "Y-yeah." "Home." "Home sweet home." "Christ." "Look, Arthur, you want my advice?" he said. Absently, he took his left hand out from between the girl's upper arm and chest wall. “Get in bed and relax." "Tell the truth." "Is it going to do you any good to sit around and stew?" "Yeah, I know." "I wouldn't even worry, for Chrissake, but, uh, you can't trust her!" "I swear to God." "I swear to God you can't.” The gray-haired man turned his head to see what the girl was doing. She had picked up the ashtray and was putting it between them. "Y-You know what I do?" "You know what I do?" "I'm ashamed to—I’m ashamed to tell ya, but you know what I very nearly goddam do every night? " "When I get home?" "You want to know?" "Arthur, listen, this isn't—" "Wait a second—I'll tell ya, God damn it." "I practically have to keep myself from opening every goddamn closet door in the apartment—I swear to God." "Every night I come home, I half expect to find a bunch of bastards hiding all over the place." "All right." "All right." Let's try to take it a little easy, Arthur," the gray-haired man said. He turned his head again toward the girl, perhaps to show her how forbearing, even stoic, his countenance was. But the girl missed seeing it. She had just overturned the ashtray with her knee and was rapidly, with her fingers, brushing the spilled ashes into a little pick-up pile. Her eyes looked up at him a second too late. "In the first place," he said into the phone, "I've told you many, many times, Arthur, that's exactly where you make your biggest mistake." "You know what you do?" "You actually go out of your way to torture yourself." " As a matter of fact, you actually inspire Joanie—" He broke off. "You're bloody lucky she's a wonderful kid." "I mean it." "You give that kid absolutely no credit for having any good taste—or brains, for Chrissake, for that matter—" "Brains?" "Jesus, if you knew how funny that was." "She thinks she's a goddamn intellectual." "That's the funny part, that's the hilarious part." "She reads the theatrical page, and she watches television till she's practically blind—so she's an intellectual." "You know who I'm married to?" "I'm married to the greatest living undeveloped, undiscovered actress, novelist, psychoanalyst, and all-around goddam unappreciated celebrity-genius in New York." "You didn't know that, did you?" "Christ, it's so funny I could cut my throat.” “In the first place,” the gray-haired man said, and slowly moved his hand and then caressed the little lock that rested on the girl’s forehead. “For a hell of an intelligent guy, you’re about as tactless as it’s humanly possible to be.” "We're mismated, that's all." "That's the whole simple story." "We're just mismated as hell." "She doesn't respect me." "She doesn't even love me, for God's sake." "She bought me a suit once." "With her own money." "I tell you about that?" "No, I—" The gray-haired man listened another moment. Then, abruptly, he turned toward the girl. The look he gave her, though only glancing, fully informed her what was suddenly going on at the other end of the phone. "Now, Arthur." "Listen." "I say this in all sincerity." "Will you get undressed and get into bed, like a good guy? " "And relax?" "Joanie'll probably be there in, in about two minutes." He listened. "Arthur?" "You hear me?" "Yeah." "I hear ya." "Listen." "I've kept you awake all night anyway." "Could I come over to your place for a drink?" "Would you mind?" The gray-haired man straightened his back and placed the flat of his free hand on the top of his head, and said, "Now, do you mean?" "Yeah, I mean if it's all right with you. "I'll only stay a minute." "I'd just like to sit down somewhere and—I don't know." "Would it be all right?" "Yeah, uh, but the point is I don't think you should, Arthur," the gray-haired man said, lowering his hand from his head. "I mean you’re, you're more than welcome to come, but I honestly think you should just sit tight and relax till Joanie waltzes in." "What, what you want to be, you want to be right there on the spot when she waltzes in." "Am I right, or not?" "Yeah." "I-I don't know, I swear to God, I don't know. "Well, I do, I honestly do," the gray-haired man said. "Look, why don't you hop in bed now, and relax, and then later, if you feel like it, give me a ring." "I mean if you feel like talking." "And don't worry." "That's the main thing." "Hear me?" "Will you do that now?" "All right." The gray-haired man continued for a moment to hold the phone to his ear, then lowered it into its cradle. "What did he say?" the girl immediately asked him. He picked his cigarette out of the ashtray—that is, selected it from an accumulation of smoked and half-smoked cigarettes. He dragged on it and said, "He wanted to come over for a drink." "God, what'd you say?" said the girl. "You heard me," the gray-haired man said, and looked at her. "You are wonderful." Absolutely marvellous," the girl said, watching him. "Well," the gray-haired man said, "it's a tough situation. I don't know how marvellous I was." "You were, you were wonderful," the girl said. The gray-haired man looked at her. "Well, it's a very, very tough situation." "The guy's obviously going through absolute—" The phone suddenly rang. The gray-haired man said "Christ!" but picked it up before the second ring. "Hello?" he said into it. "Lee, were you asleep?" "No, uh, no." "Listen, I just thought you'd want to know, Joanie just barged in." "What?" said the gray-haired man, and bridged his left hand over his eyes, though the light was behind him. "Yeah, she, she just barged in, about ten seconds after I spoke to you." "I just thought I'd give you a ring while she's in the john." "Listen, thanks a million, Lee." "I mean it." "You weren't asleep, were ya?" "No, no, I was just—No, no," the gray-haired man said, leaving his fingers bridged over his eyes. He cleared his throat. "Yeah, what, uh, what happened was, apparently Leona got stinking and, uh, then had a goddam crying jag, and Bob wanted Joanie to, to go out and grab a drink with them somewhere. "Anyway, so she's home." "What a rat race." "Honest to God, I think it's this goddam New York." "What I think maybe we'll do, if everything goes along all right, we'll get ourselves a little place in Connecticut maybe." "Not far out, necessarily, but far enough that we can lead a normal goddam life." "I mean she's crazy about plants and all that stuff." "She'd probably go mad if she had a garden, if she had her own goddam garden and stuff." "Know what I mean?" "I mean—except you—who do we know in New York except a bunch of neurotics?" "It's bound to undermine even a normal person sooner or later." "Know what I mean?" "Listen, Arthur," the gray-haired man interrupted, taking his hand away from his face, "I have a hell of a headache all of a sudden. I don't know where I got the, the bloody thing from." "You mind if we cut this short?" "I'll talk to you in the morning—all right?" He listened for another moment, then hung up. Again the girl immediately spoke to him, but he didn't answer her. He picked a burning cigarette—the girl's—out of the ashtray and started to bring it to his mouth, but it slipped out of his fingers. The girl tried to help him retrieve it before anything was burned, but he told her to just sit still, for Chrissake, and she pulled back her hand."""
    cleaned_story = """"The phone rings and interrupts the gray-haired man and the girl in bed. The gray-haired man answers the phone and speaks to Arthur. Arthur asks the gray-haired man whether he saw Joanie leave the party. The gray-haired man tells Arthur he did not see Joanie leave. Arthur reveals that Joanie has gone missing after the party. Arthur expresses his deep frustration and exhaustion with the marriage. The gray-haired man signals the girl for a cigarette during the call. The gray-haired man advises Arthur to get into bed and wait for Joanie. Arthur confesses that he searches the apartment closets for men every night. The gray-haired man tells Arthur that his paranoia actively drives Joanie away. Arthur bitterly mocks Joanie's intellectual pretensions to the gray-haired man. Arthur asks the gray-haired man if he can come over for a drink. The gray-haired man discourages Arthur from coming and ends the call. The girl praises the gray-haired man for how he handled Arthur. The phone rings again almost immediately after they hang up. Arthur tells the gray-haired man that Joanie has just arrived home. Arthur explains that Leona had a crying jag and delayed Joanie's return. Arthur tells the gray-haired man that he plans to move the family to Connecticut. The gray-haired man cuts the call short by claiming a sudden headache. The gray-haired man drops the girl's cigarette and refuses her help retrieving it."""

    topics = [ # cheating, paranoid
        "A man unknowingly calls his wife's lover to report her missing.",
        "A paranoid husband calls his exhausted friend in the middle of the night, convinced without cause that his wife is being unfaithful."
    ]

    output = calculate_sequentiality(model=model_id,
                        history_lengths=[2, 4, 6, 8], # max history (207 sentences in the raw story)
                        text_input=[raw_story, raw_story, cleaned_story, cleaned_story],
                        topics=topics * 2)

    # Save to outputs folder
    os.makedirs("./outputs/ensemble/phone_call_clean/", exist_ok=True)
    
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./outputs/ensemble/phone_call_clean/{safe_model_name}.csv")

# Example usage:
if __name__ == "__main__":
    # This is the function to use when running on hpc - see documentation for parameters
    # run_sequential(int(sys.argv[1]))
    
    # if len(sys.argv) < 2:
    #     print("Usage: python main.py <model_index>")
    #     print(f"Available models (0-{len(MODEL_IDS)-1}):")
    #     for i, model in enumerate(MODEL_IDS):
    #         print(f"  {i}: {model}")
    #     sys.exit(1)
    
    # idx = int(sys.argv[1])
    # generate_model_sequentiality(idx)

    model_idx = int(sys.argv[1])
    if model_idx in range(len(MODEL_IDS)):
        model = MODEL_IDS[model_idx]
    else:
        print("invalid model index")
        exit(-2)

    phone_call_cleaned(model)
