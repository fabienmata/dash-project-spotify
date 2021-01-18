import pandas as pd 
import json
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials

# project's credentials obtained in spotify's developer dashboard
cid = "ff018bf33fe84021af42bd7e948e4ce2"
secret =  "b92ead7be9894651aeb674f3496427b8"

# as spotify want us to use the credential each time we call a request, the call in a simpler way : 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

#a function to request informations for each song in the playlist
def analyze_playlist(creator, playlist_id):

# Create the columns name of the future dataframe
    playlist_features_list = ["artist","album","track_name", "track_id","genres","danceability","energy","key","loudness",   "mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]

    playlist_df = pd.DataFrame(columns = playlist_features_list)

    my_songs = sp.user_playlist_tracks(creator, playlist_id)

#function to obtain all the songs instead of just 100 (which is the default)
    def get_all_the_songs_in_the_playlist(tracks):
        playlist = tracks["items"]
        while tracks['next']:
            tracks = sp.next(tracks)
            playlist.extend(tracks['items'])
        return playlist
    playlist = get_all_the_songs_in_the_playlist(my_songs)

# Loop through every track in the playlist, extract features and append the features to the playlist df
    for track in playlist: # Create empty dict
        playlist_features = {} # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]

#get the genres of the artist      
        artist_id = track['track']['album']['artists'][0]['id']
        artist = sp.artist(artist_id)
        playlist_features["genres"] = ', '.join(artist['genres'][:3])

# Get audio features for the track
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[5:]:
            playlist_features[feature] = audio_features[feature]

# Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)

    return playlist_df