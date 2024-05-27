    
def search(youtube_video_url, query):

    from sentence_transformers import SentenceTransformer, util
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    import pandas as pd
    import numpy as np
    import os
    from pytube import YouTube
    import whisper
    import pandas as pd

    # youtube_video_url = "https://www.youtube.com/watch?v=k7aQEqDbuf8" 
    # query = input()


    model_name = 'D:/Projects/FYP/fyp_web_app/searching_model' 
    # model_name = 'D:/Projects/FYP/fyp_web_app/searching_model' # give the location to the model here
    sbert_model = SentenceTransformer(model_name)

    # function to extarct video subs
    def extract_youtube_subtitles(video_url):
        # Extract the video ID from the YouTube video URL
        video_id = video_url.split("v=")[1] 
        try:
            # Get the transcript for the video
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            # Extract timestamps, subtitles, and end times into separate lists
            start_times = [entry['start'] for entry in transcript]
            end_times = [entry['start'] + entry['duration'] for entry in transcript]
            subtitles = [entry['text'] for entry in transcript]
            # Create a DataFrame
            df = pd.DataFrame({
                'Start Time Stamp': start_times,
                'End Time Stamp': end_times,
                'Subtitle': subtitles
            })
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None
    # this extrcats and returns subs
    
    # put the check subs function here so that we would know what kind of subs we are extracting 
    def check_subs(video_url):
        # List available transcripts
        video_id = video_url.split('v=')[-1]
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            if transcript_list:
                print("it has subs")
        except TranscriptsDisabled: 
            print("video doesnt have subs") 
            return "no" 
            
        print("error before coming here")
        
        # Check if manually created English subs are available
        manual = False 
        auto = False 

        try:
            transcript = transcript_list.find_manually_created_transcript(['en']) 
            if transcript: 
                print("Manually created English subs are available.")  
                manual = True
        except NoTranscriptFound: 
            print("NoTranscriptFound for manual")  

        try:
            transcript = transcript_list.find_generated_transcript(['en'])
            if transcript:
                print("Auto created English subs are available.")
                auto = True 
        except NoTranscriptFound:
            print("NoTranscriptFound for auto") 
        
        if manual:
            return "manual"
        elif auto:
            return "auto"
        else: 
            return "no"
       
    def make_subsDF_using_whisperAI(url,download_path):

        def download_youtube_audio(url, download_path):
            try:
                video = YouTube(url)
                # print("Video:", video)
                # print("Type of video:", type(video))
                
                audio_stream = video.streams.filter(only_audio=True).first()
                # print("Audio stream:", audio_stream)
                # print("Type of audio stream:", type(audio_stream))
                
                audio_file = audio_stream.download(output_path=download_path)
                # print("Audio file:", audio_file)
                # print("Type of audio file:", type(audio_file))
                
                base, ext = os.path.splitext(audio_file)
                # print("Base and extension:", base, ext)
                
                new_file = base + '.wav'
                os.rename(audio_file, new_file)
                
                # print(f"Audio downloaded: {video.title}.wav")
                # print(f"Download path: {download_path}")
                
                return new_file
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return None

        def generate_subtitles(audio_file):
            try:
                model = whisper.load_model("base")
                result = model.transcribe(audio_file)
                segments = result["segments"]
                
                subtitles_data = []
                for segment in segments:
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    subtitle = segment["text"].strip()
                    subtitles_data.append([start_time, end_time, subtitle])
                
                subtitle_df = pd.DataFrame(subtitles_data, columns=["Start Time Stamp", "End Time Stamp", "Subtitle"])
                print("Subtitle DataFrame:")
                print(subtitle_df)
                
                return subtitle_df
            except Exception as e:
                print(f"An error occurred during subtitle generation: {str(e)}")
                return None

        def format_timestamp(seconds):
            minutes, seconds = divmod(int(seconds), 60)
            hours, minutes = divmod(minutes, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Download YouTube audio

        audio_file = download_youtube_audio(url, download_path)

        if audio_file:   
            subtitle_df = generate_subtitles(audio_file)
            if subtitle_df is not None: 
                print("created subtitles df using whisperAI") 
                return subtitle_df 
            else: 
                print("subtitles df was returned None (subs were not formed)") 
        else: 
            print("something went wrong with getting audio file using pytube!") 

    def convert_timestamps_to_readable_format(df):
        # Ensure the DataFrame has "start_time" and "end_time" columns
        if "Start Time Stamp" not in df.columns or "End Time Stamp" not in df.columns:
            raise ValueError("DataFrame must contain 'Start Time Stamp' and 'End Time Stamp' columns.")

        # Convert timestamps to readable format
        df["Start Time Stamp"] = pd.to_timedelta(df["Start Time Stamp"], unit='s').dt.total_seconds().astype(int).apply(lambda x: str(pd.to_datetime(x, unit='s').time()))
        df["End Time Stamp"] = pd.to_timedelta(df["End Time Stamp"], unit='s').dt.total_seconds().astype(int).apply(lambda x: str(pd.to_datetime(x, unit='s').time()))

        return df


    # Convert timestamps to readable format
    # top_10_df["start_time"] = pd.to_timedelta(top_10_df["start_time"], unit='s').dt.total_seconds().astype(int).apply(lambda x: str(pd.to_datetime(x, unit='s').time()))
    # top_10_df["end_time"] = pd.to_timedelta(top_10_df["end_time"], unit='s').dt.total_seconds().astype(int).apply(lambda x: str(pd.to_datetime(x, unit='s').time()))
    # print(time_formatted_df)

    def merge_k_subtitles(subtitle_df, k):
        # Ensure k is a positive integer
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        # Create a list to store DataFrames for concatenation
        dfs_to_concat = []
        # Iterate through the original DataFrame and merge every k rows
        for i in range(0, len(subtitle_df)-k, k):
            start_time = subtitle_df.loc[i, "Start Time Stamp"]
            end_time = subtitle_df.loc[i + k - 1, "End Time Stamp"]
            merged_subtitle = " ".join(subtitle_df.loc[i:i + k - 1, "Subtitle"])
            # Create a DataFrame for the merged subtitle
            merged_df = pd.DataFrame({"Start Time Stamp": [start_time],
                                    "End Time Stamp": [end_time],
                                    "Subtitle": [merged_subtitle]})
            dfs_to_concat.append(merged_df)
        # Concatenate the list of DataFrames into a single DataFrame
        merged_df = pd.concat(dfs_to_concat, ignore_index=True)
        return merged_df
    # so how do i choose k now? ill choose based on average subtitle length of each sentence, 
    #first lets find the k value
    def average_subtitle_length(df):
        # Calculate the length of each subtitle
        df['Subtitle Length'] = df['Subtitle'].apply(lambda x: len(x.split()))
        # print(df[['Subtitle','Subtitle Length']])
        # Calculate the average length of subtitles
        average_length = df['Subtitle Length'].mean()
        return average_length

    def findk(df):
        avg = average_subtitle_length(df) 
        k = 2 
        if avg <= 4: 
            k = 4 
        elif avg >4 and avg <8: 
            k = 3 
        else: 
            k = 2 
        return k 
    
    subs_type = check_subs(youtube_video_url) 
    if subs_type == "manual":
        subtitle_df = extract_youtube_subtitles(youtube_video_url) 
        # print(subtitle_df)
    elif subs_type == "auto":
        subtitle_df = extract_youtube_subtitles(youtube_video_url) 
    else:
        # make a function that uses whisperAI to create subs 
        audio_location = "subtitles_store_if_making_them"
        subtitle_df = make_subsDF_using_whisperAI(youtube_video_url,audio_location)
        # print(subtitle_df)

    
    print(subtitle_df)
    if subs_type == "manual" or subs_type == "auto":
        convert_timestamps_to_readable_format(subtitle_df)

    k = findk(subtitle_df) 
    print("average length of the subtitles is",k) 
    k_merged_df = merge_k_subtitles(subtitle_df, k)
    k_merged_df['Subtitle'] = k_merged_df['Subtitle'].str.replace('\n', ' ')
    # print(k_merged_df[:5])
    # convert df to list
    subtitles_list = k_merged_df[["Start Time Stamp", "Subtitle"]].values.tolist()
    print(subtitles_list)
    # create embeddings
    subtitle_embeddings = sbert_model.encode([entry[1] for entry in subtitles_list], convert_to_tensor=True)
    # print(query,type(query))
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, subtitle_embeddings)
    top_indices = np.argsort(similarities.numpy(), axis=1)[:, ::-1][:, :10].tolist() # sorting and then returning top 10
    print("top indices",top_indices) 
    top_10_similar_subtitles = [subtitles_list[i] for i in top_indices[0]]
    print(top_10_similar_subtitles)
    # Create a DataFrame for the top 5 similar subtitles
    top_10_df = pd.DataFrame(top_10_similar_subtitles, columns=["start_time","Subtitle"])
    print(top_10_df) 

    def convert_dataframe_to_list(df):
        result = []
        for _, row in df.iterrows():
            start_time = row['start_time']
            subtitle = row['Subtitle']
            # Create the formatted string
            formatted_string = f"{start_time}  ->  {subtitle}"
            result.append(formatted_string)
        return result
    ans = convert_dataframe_to_list(top_10_df) 
    return ans
    
# ans = search("https://www.youtube.com/watch?v=HgRhJCOdvpI","nice grappling exchange")
# print(ans)
# ans = search("https://www.youtube.com/watch?v=OiKQpnWndOg","thermals of the device") 
# print(ans)
# ans = search("https://www.youtube.com/watch?v=Cpdb__QWEiw","fighting men with guns")
# print(ans)
# print(ans) 












