# 11 domains
domain_types = ["Alarm",
                "Calling",
                "Event",
                "Messaging",
                "Music",
                "News",
                "People",
                "Recipes",
                "Reminder",
                "Timer",
                "Weather"]

# 117 intent types => actual only 14 intent types over each languages and each splits
intent_types = ["alarm:CREATE_ALARM",
                "alarm:DELETE_ALARM",
                "alarm:GET_ALARM",
                "alarm:SILENCE_ALARM",
                "alarm:SNOOZE_ALARM",
                "alarm:UPDATE_ALARM",
                "calling:ANSWER_CALL",
                "calling:CANCEL_CALL",
                "calling:CREATE_CALL",
                "calling:END_CALL",
                "calling:GET_AVAILABILITY",
                "calling:GET_CALL",
                "calling:GET_CALL_CONTACT",
                "calling:GET_CALL_TIME",
                "calling:HOLD_CALL",
                "calling:IGNORE_CALL",
                "calling:MERGE_CALL",
                "calling:RESUME_CALL",
                "calling:SET_AVAILABLE",
                "calling:SET_DEFAULT_PROVIDER_CALLING",
                "calling:SET_UNAVAILABLE",
                "calling:SWITCH_CALL",
                "calling:UPDATE_CALL",
                "calling:UPDATE_METHOD_CALL",
                "event:DISPREFER",
                "event:GET_ATTENDEE_EVENT",
                "event:GET_CATEGORY_EVENT",
                "event:GET_DATE_TIME_EVENT",
                "event:GET_EVENT",
                "event:GET_LOCATION",
                "event:PREFER",
                "event:SET_RSVP_INTERESTED",
                "event:SET_RSVP_NO",
                "event:SET_RSVP_YES",
                "event:SHARE_EVENT",
                "messaging:CANCEL_MESSAGE",
                "messaging:GET_GROUP",
                "messaging:GET_MESSAGE",
                "messaging:GET_MESSAGE_CONTACT",
                "messaging:SEND_MESSAGE",
                "music:ADD_TO_PLAYLIST_MUSIC",
                "music:CREATE_PLAYLIST_MUSIC",
                "music:DELETE_PLAYLIST_MUSIC",
                "music:DISLIKE_MUSIC",
                "music:FAST_FORWARD_MUSIC",
                "music:FOLLOW_MUSIC",
                "music:GET_LYRICS_MUSIC",
                "music:GET_TRACK_INFO_MUSIC",
                "music:LIKE_MUSIC",
                "music:LOOP_MUSIC",
                "music:PAUSE_MUSIC",
                "music:PLAY_MEDIA",
                "music:PLAY_MUSIC",
                "music:PREVIOUS_TRACK_MUSIC",
                "music:QUESTION_MUSIC",
                "music:REMOVE_FROM_PLAYLIST_MUSIC",
                "music:REPEAT_ALL_MUSIC",
                "music:REPEAT_ALL_OFF_MUSIC",
                "music:REPLAY_MUSIC",
                "music:RESUME_MUSIC",
                "music:REWIND_MUSIC",
                "music:SET_DEFAULT_PROVIDER_MUSIC",
                "music:SKIP_TRACK_MUSIC",
                "music:START_SHUFFLE_MUSIC",
                "music:STOP_MUSIC",
                "music:STOP_SHUFFLE_MUSIC",
                "music:UNLOOP_MUSIC",
                "news:GET_DETAILS_NEWS",
                "news:GET_STORIES_NEWS",
                "news:QUESTION_NEWS",
                "people:GET_AGE",
                "people:GET_CONTACT",
                "people:GET_CONTACT_METHOD",
                "people:GET_EDUCATION_DEGREE",
                "people:GET_EDUCATION_TIME",
                "people:GET_EMPLOYER",
                "people:GET_EMPLOYMENT_TIME",
                "people:GET_GENDER",
                "people:GET_INFO_CONTACT",
                "people:GET_JOB",
                "people:GET_LANGUAGE",
                "people:GET_LIFE_EVENT",
                "people:GET_LIFE_EVENT_TIME",
                "people:GET_LOCATION",
                "people:GET_MAJOR",
                "people:GET_MUTUAL_FRIENDS",
                "people:GET_UNDERGRAD",
                "recipes:GET_INFO_RECIPES",
                "recipes:GET_RECIPES",
                "recipes:IS_TRUE_RECIPES",
                "reminder:CREATE_REMINDER",
                "reminder:DELETE_REMINDER",
                "reminder:GET_REMINDER",
                "reminder:GET_REMINDER_AMOUNT",
                "reminder:GET_REMINDER_DATE_TIME",
                "reminder:GET_REMINDER_LOCATION",
                "reminder:HELP_REMINDER",
                "reminder:UPDATE_REMINDER",
                "reminder:UPDATE_REMINDER_DATE_TIME",
                "reminder:UPDATE_REMINDER_LOCATION",
                "reminder:UPDATE_REMINDER_TODO",
                "timer:ADD_TIME_TIMER",
                "timer:CREATE_TIMER",
                "timer:DELETE_TIMER",
                "timer:GET_TIMER",
                "timer:PAUSE_TIMER",
                "timer:RESTART_TIMER",
                "timer:RESUME_TIMER",
                "timer:SUBTRACT_TIME_TIMER",
                "timer:UPDATE_TIMER",
                "weather:GET_AIRQUALITY",
                "weather:GET_SUNRISE",
                "weather:GET_SUNSET",
                "weather:GET_WEATHER"]

# 78 slot types
slot_types = ["O",
              "X",
              "B-AGE",
              "I-AGE",
              "B-ALARM_NAME",
              "I-ALARM_NAME",
              "B-AMOUNT",
              "I-AMOUNT",
              "B-ATTENDEE",
              "I-ATTENDEE",
              "B-ATTENDEE_EVENT",
              "I-ATTENDEE_EVENT",
              "B-ATTRIBUTE_EVENT",
              "I-ATTRIBUTE_EVENT",
              "B-CATEGORY_EVENT",
              "I-CATEGORY_EVENT",
              "B-CONTACT",
              "I-CONTACT",
              "B-CONTACT_ADDED",
              "I-CONTACT_ADDED",
              "B-CONTACT_METHOD",
              "I-CONTACT_METHOD",
              "B-CONTACT_RELATED",
              "I-CONTACT_RELATED",
              "B-CONTACT_REMOVED",
              "I-CONTACT_REMOVED",
              "B-CONTENT_EXACT",
              "I-CONTENT_EXACT",
              "B-DATE_TIME",
              "I-DATE_TIME",
              "B-EDUCATION_DEGREE",
              "I-EDUCATION_DEGREE",
              "B-EMPLOYER",
              "I-EMPLOYER",
              "B-GENDER",
              "I-GENDER",
              "B-GROUP",
              "I-GROUP",
              "B-JOB",
              "I-JOB",
              "B-LIFE_EVENT",
              "I-LIFE_EVENT",
              "B-LOCATION",
              "I-LOCATION",
              "B-MAJOR",
              "I-MAJOR",
              "B-METHOD_RECIPES",
              "I-METHOD_RECIPES",
              "B-METHOD_RETRIEVAL_REMINDER",
              "I-METHOD_RETRIEVAL_REMINDER",
              "B-METHOD_TIMER",
              "I-METHOD_TIMER",
              "B-MUSIC_ALBUM_MODIFIER",
              "I-MUSIC_ALBUM_MODIFIER",
              "B-MUSIC_ALBUM_TITLE",
              "I-MUSIC_ALBUM_TITLE",
              "B-MUSIC_ARTIST_NAME",
              "I-MUSIC_ARTIST_NAME",
              "B-MUSIC_GENRE",
              "I-MUSIC_GENRE",
              "B-MUSIC_PLAYLIST_MODIFIER",
              "I-MUSIC_PLAYLIST_MODIFIER",
              "B-MUSIC_PLAYLIST_TITLE",
              "I-MUSIC_PLAYLIST_TITLE",
              "B-MUSIC_PROVIDER_NAME",
              "I-MUSIC_PROVIDER_NAME",
              "B-MUSIC_RADIO_ID",
              "I-MUSIC_RADIO_ID",
              "B-MUSIC_REWIND_TIME",
              "I-MUSIC_REWIND_TIME",
              "B-MUSIC_TRACK_TITLE",
              "I-MUSIC_TRACK_TITLE",
              "B-MUSIC_TYPE",
              "I-MUSIC_TYPE",
              "B-NAME_APP",
              "I-NAME_APP",
              "B-NEWS_CATEGORY",
              "I-NEWS_CATEGORY",
              "B-NEWS_REFERENCE",
              "I-NEWS_REFERENCE",
              "B-NEWS_SOURCE",
              "I-NEWS_SOURCE",
              "B-NEWS_TOPIC",
              "I-NEWS_TOPIC",
              "B-NEWS_TYPE",
              "I-NEWS_TYPE",
              "B-ORDINAL",
              "I-ORDINAL",
              "B-PERIOD",
              "I-PERIOD",
              "B-PERSON_REMINDED",
              "I-PERSON_REMINDED",
              "B-PHONE_NUMBER",
              "I-PHONE_NUMBER",
              "B-RECIPES_ATTRIBUTE",
              "I-RECIPES_ATTRIBUTE",
              "B-RECIPES_COOKING_METHOD",
              "I-RECIPES_COOKING_METHOD",
              "B-RECIPES_CUISINE",
              "I-RECIPES_CUISINE",
              "B-RECIPES_DIET",
              "I-RECIPES_DIET",
              "B-RECIPES_DISH",
              "I-RECIPES_DISH",
              "B-RECIPES_EXCLUDED_INGREDIENT",
              "I-RECIPES_EXCLUDED_INGREDIENT",
              "B-RECIPES_INCLUDED_INGREDIENT",
              "I-RECIPES_INCLUDED_INGREDIENT",
              "B-RECIPES_MEAL",
              "I-RECIPES_MEAL",
              "B-RECIPES_QUALIFIER_NUTRITION",
              "I-RECIPES_QUALIFIER_NUTRITION",
              "B-RECIPES_RATING",
              "I-RECIPES_RATING",
              "B-RECIPES_SOURCE",
              "I-RECIPES_SOURCE",
              "B-RECIPES_TIME_PREPARATION",
              "I-RECIPES_TIME_PREPARATION",
              "B-RECIPES_TYPE",
              "I-RECIPES_TYPE",
              "B-RECIPES_TYPE_NUTRITION",
              "I-RECIPES_TYPE_NUTRITION",
              "B-RECIPES_UNIT_MEASUREMENT",
              "I-RECIPES_UNIT_MEASUREMENT",
              "B-RECIPES_UNIT_NUTRITION",
              "I-RECIPES_UNIT_NUTRITION",
              "B-RECIPIENT",
              "I-RECIPIENT",
              "B-SCHOOL",
              "I-SCHOOL",
              "B-SENDER",
              "I-SENDER",
              "B-SIMILARITY",
              "I-SIMILARITY",
              "B-TIMER_NAME",
              "I-TIMER_NAME",
              "B-TITLE_EVENT",
              "I-TITLE_EVENT",
              "B-TODO",
              "I-TODO",
              "B-TYPE_CONTACT",
              "I-TYPE_CONTACT",
              "B-TYPE_CONTENT",
              "I-TYPE_CONTENT",
              "B-TYPE_RELATION",
              "I-TYPE_RELATION",
              "B-USER_ATTENDEE_EVENT",
              "I-USER_ATTENDEE_EVENT",
              "B-WEATHER_ATTRIBUTE",
              "I-WEATHER_ATTRIBUTE",
              "B-WEATHER_TEMPERATURE_UNIT",
              "I-WEATHER_TEMPERATURE_UNIT"]
