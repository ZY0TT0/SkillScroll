import streamlit as st
import pandas as pd
import os
from datetime import datetime
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_main_tag(tags, topic):
    if isinstance(tags, str) and tags.strip():
        return tags.split(",")[0].strip()
    return str(topic) if isinstance(topic, str) else "this concept"

def generate_summary_offline(title, topic, tags, level):
    main_tag = get_main_tag(tags, topic)
    level = (level or "Beginner").strip()

    if level.lower() == "beginner":
        return f"This reel gives a simple introduction to {main_tag} in the context of {topic}."
    elif level.lower() == "intermediate":
        return f"This reel explains important details of {main_tag} to deepen your understanding of {topic}."
    else:
        return f"This reel focuses on advanced aspects of {main_tag} and how it is used in real problems related to {topic}."

def generate_quiz_offline(title, topic, tags, level):
    main_tag = get_main_tag(tags, topic)
    level = (level or "Beginner").strip()

    questions = []

    q1 = {
        "q": "What is the main concept explained in this reel?",
        "options": [main_tag, "Cricket", "Cooking", "Random vlog"],
        "answer": main_tag,
    }
    questions.append(q1)

    q2 = {
        "q": f"What is the main goal of this reel about {main_tag}?",
        "options": [
            "To entertain only",
            "To help you understand a concept",
            "To show travel vlogs",
            "To review a product",
        ],
        "answer": "To help you understand a concept",
    }
    questions.append(q2)

    q3 = {
        "q": "This reel is primarily targeted at which level of learner?",
        "options": ["Beginner", "Intermediate", "Advanced"],
        "answer": "Beginner" if level.lower() == "beginner"
        else "Intermediate" if level.lower() == "intermediate"
        else "Advanced",
    }
    questions.append(q3)

    return questions

def generate_practice_offline(title, topic, tags, level):
    main_tag = get_main_tag(tags, topic)
    return (
        f"In 3–4 lines, explain {main_tag} in your own words and give one simple example "
        f"from the topic {topic}."
    )

VIDEO_CSV = "videos.csv"
COURSE_CSV = "courses.csv"
PROGRESS_CSV = "progress.csv"
COMMENTS_CSV = "comments.csv"
POLLS_CSV = "polls.csv"
POLL_VOTES_CSV = "poll_votes.csv"

def load_or_create_csv(path, columns):
    if os.path.exists(path):
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                df[col] = None
        return df[columns]
    else:
        return pd.DataFrame(columns=columns)

def load_data():
    videos = load_or_create_csv(
        VIDEO_CSV,
        ["video_id", "title", "topic", "concept_tags", "level",
         "course_id", "url", "duration_sec"]
    )

    courses = load_or_create_csv(
        COURSE_CSV,
        ["course_id", "course_name", "description", "level", "topic"]
    )

    progress = load_or_create_csv(
        PROGRESS_CSV,
        ["user_id", "course_id", "video_id", "watched"]
    )

    comments = load_or_create_csv(
        COMMENTS_CSV,
        ["video_id", "user_id", "text", "is_creator", "timestamp"]
    )

    polls = load_or_create_csv(
        POLLS_CSV,
        ["poll_id", "video_id", "question",
         "option_1", "option_2", "option_3", "option_4"]
    )

    poll_votes = load_or_create_csv(
        POLL_VOTES_CSV,
        ["poll_id", "user_id", "chosen_option"]
    )

    return videos, courses, progress, comments, polls, poll_votes

def save_data(videos, courses, progress, comments, polls, poll_votes):
    videos.to_csv(VIDEO_CSV, index=False)
    courses.to_csv(COURSE_CSV, index=False)
    progress.to_csv(PROGRESS_CSV, index=False)
    comments.to_csv(COMMENTS_CSV, index=False)
    polls.to_csv(POLLS_CSV, index=False)
    poll_votes.to_csv(POLL_VOTES_CSV, index=False)

def get_current_user():
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"
    return st.session_state.user_id

def build_recommendations(videos, progress, user_id, top_n=10):
    if videos.empty:
        return videos

    texts = (
        videos["title"].fillna("") + " " +
        videos["topic"].fillna("") + " " +
        videos["concept_tags"].fillna("")
    )

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    user_watched = progress[
        (progress["user_id"] == user_id) & (progress["watched"] == 1)
    ]

    if len(user_watched["video_id"].unique()) < 2:
        return videos

    watched_ids = user_watched["video_id"].unique().tolist()
    watched_idx = videos[videos["video_id"].isin(watched_ids)].index.tolist()

    if not watched_idx:
        return videos

    watched_vectors = tfidf_matrix[watched_idx]
    user_profile_vec = watched_vectors.mean(axis=0)
    user_profile_vec = np.asarray(user_profile_vec).reshape(1, -1)

    sims = cosine_similarity(user_profile_vec, tfidf_matrix).ravel()

    videos_copy = videos.copy()
    videos_copy["similarity"] = sims

    videos_copy = videos_copy[~videos_copy["video_id"].isin(watched_ids)]
    videos_copy = videos_copy[videos_copy["similarity"] > 0]

    videos_copy = videos_copy.sort_values("similarity", ascending=False)

    return videos_copy.head(top_n)

st.set_page_config(page_title="SkillScroll – Bite-Sized Learning", layout="wide")
st.title("SkillScroll – Bite-Sized Learning App")

videos, courses, progress, comments, polls, poll_votes = load_data()
user_id = get_current_user()

st.sidebar.header("Navigation")
mode = st.sidebar.radio("Mode", ["Learn", "My Courses", "Creator Studio"])
st.sidebar.write(f"Logged in as: `{user_id}` (demo)")

if mode == "Learn":
    st.subheader("🎓 Personalized Learning Feed")

    topics = ["All"] + sorted([t for t in videos["topic"].dropna().unique().tolist()])
    levels = ["All"] + sorted([l for l in videos["level"].dropna().unique().tolist()])

    col1, col2, col3 = st.columns(3)
    with col1:
        topic_filter = st.selectbox("Filter by Topic", topics)
    with col2:
        level_filter = st.selectbox("Filter by Level", levels)
    with col3:
        feed_type = st.selectbox("Feed Type", ["Recommended For You (ML)", "All Videos"])

    if videos.empty:
        st.info("No videos yet. Ask a creator to upload from the Creator Studio.")
    else:
        if feed_type == "Recommended For You (ML)":
            feed = build_recommendations(videos, progress, user_id, top_n=20)
            if feed.empty:
                feed = videos.copy()
        else:
            feed = videos.copy()

        if topic_filter != "All":
            feed = feed[feed["topic"] == topic_filter]
        if level_filter != "All":
            feed = feed[feed["level"] == level_filter]

        if feed.empty:
            st.warning("No videos match the current filters.")
        else:
            for _, row in feed.iterrows():
                st.markdown("---")
                st.markdown(f"### {row['title']}")
                st.caption(
                    f"Topic: {row['topic']} | Level: {row['level']} | "
                    f"{int(row['duration_sec']) if pd.notna(row['duration_sec']) else '--'} sec"
                )
                st.caption(f"Tags: {row['concept_tags'] if pd.notna(row['concept_tags']) else '-'}")

                url = str(row["url"]).strip() if pd.notna(row["url"]) else ""
                if url:
                    if "youtube.com/shorts/" in url:
                        try:
                            vid_id = url.split("shorts/")[1].split("?")[0].split("/")[0]
                            url = f"https://www.youtube.com/watch?v={vid_id}"
                        except Exception:
                            pass
                    st.video(url)

                c1, c2 = st.columns(2)
                if c1.button("✅ Mark as Watched", key=f"watch_{row['video_id']}"):
                    new_row = {
                        "user_id": user_id,
                        "course_id": row["course_id"],
                        "video_id": row["video_id"],
                        "watched": 1,
                    }
                    progress = pd.concat([progress, pd.DataFrame([new_row])], ignore_index=True)
                    save_data(videos, courses, progress, comments, polls, poll_votes)
                    st.success("Marked as watched.")

                if c2.button("📚 Go to Course", key=f"goto_{row['video_id']}"):
                    course_id = row["course_id"]
                    if pd.isna(course_id) or course_id not in courses["course_id"].astype(str).tolist():
                        st.warning("This reel is not linked to any micro-course.")
                    else:
                        course_row = courses[courses["course_id"] == course_id].iloc[0]
                        st.info(
                            f"Micro-course: **{course_row['course_name']}** "
                            f"(Topic: {course_row['topic']}, Level: {course_row['level']})"
                        )
                        course_videos = videos[videos["course_id"] == course_id]
                        for _, v in course_videos.iterrows():
                            st.markdown(f"- {v['title']} ({v['duration_sec']} sec)")

                st.markdown("**💬 Comments & Q&A**")
                video_comments = comments[comments["video_id"] == row["video_id"]]

                if video_comments.empty:
                    st.caption("No comments yet. Be the first to ask a question!")
                else:
                    for _, c in video_comments.sort_values("timestamp").iterrows():
                        prefix = "Creator" if str(c["is_creator"]) == "1" else "Learner"
                        st.markdown(f"- **{prefix}:** {c['text']}")

                new_comment = st.text_input(
                    "Add a comment or question",
                    key=f"comment_{row['video_id']}"
                )
                if st.button("Post Comment", key=f"post_{row['video_id']}"):
                    if new_comment.strip():
                        new_row = {
                            "video_id": row["video_id"],
                            "user_id": user_id,
                            "text": new_comment.strip(),
                            "is_creator": 0,
                            "timestamp": datetime.now().isoformat(),
                        }
                        comments = pd.concat([comments, pd.DataFrame([new_row])], ignore_index=True)
                        save_data(videos, courses, progress, comments, polls, poll_votes)
                        st.success("Comment posted.")
                    else:
                        st.error("Comment cannot be empty.")

                st.markdown("**📊 Quick Poll**")
                poll_row = polls[polls["video_id"] == row["video_id"]]
                if poll_row.empty:
                    st.caption("No poll for this reel.")
                else:
                    poll_row = poll_row.iloc[0]
                    options = []
                    labels = []
                    if isinstance(poll_row["option_1"], str) and poll_row["option_1"]:
                        options.append("option_1")
                        labels.append(poll_row["option_1"])
                    if isinstance(poll_row["option_2"], str) and poll_row["option_2"]:
                        options.append("option_2")
                        labels.append(poll_row["option_2"])
                    if isinstance(poll_row["option_3"], str) and poll_row["option_3"]:
                        options.append("option_3")
                        labels.append(poll_row["option_3"])
                    if isinstance(poll_row["option_4"], str) and poll_row["option_4"]:
                        options.append("option_4")
                        labels.append(poll_row["option_4"])

                    if options:
                        user_prev_vote = poll_votes[
                            (poll_votes["poll_id"] == poll_row["poll_id"]) &
                            (poll_votes["user_id"] == user_id)
                        ]
                        if not user_prev_vote.empty:
                            st.caption(f"You already voted: {user_prev_vote.iloc[0]['chosen_option']}")
                        else:
                            choice = st.radio(
                                poll_row["question"],
                                labels,
                                key=f"poll_{poll_row['poll_id']}"
                            )
                            if st.button("Submit Vote", key=f"vote_{poll_row['poll_id']}"):
                                new_vote = {
                                    "poll_id": poll_row["poll_id"],
                                    "user_id": user_id,
                                    "chosen_option": choice,
                                }
                                poll_votes = pd.concat(
                                    [poll_votes, pd.DataFrame([new_vote])],
                                    ignore_index=True
                                )
                                save_data(videos, courses, progress, comments, polls, poll_votes)
                                st.success("Thanks for voting!")
                    else:
                        st.caption("Poll has no options configured.")

                                # ---------- AI Learning Assistant (optional) ----------
                with st.expander("🧠 AI Learning Assistant"):
                    summary = generate_summary_offline(
                        row["title"], row["topic"], row["concept_tags"], row["level"]
                    )
                    st.markdown(f"**Summary:** {summary}")

                    quiz = generate_quiz_offline(
                        row["title"], row["topic"], row["concept_tags"], row["level"]
                    )
                    st.markdown("**Quick Quiz:**")
                    for i, q in enumerate(quiz):
                        st.markdown(f"**Q{i+1}. {q['q']}**")
                        user_choice = st.radio(
                            "Choose an option:",
                            q["options"],
                            key=f"quiz_{row['video_id']}_{i}",
                        )
                        if st.button("Check", key=f"check_{row['video_id']}_{i}"):
                            if user_choice == q["answer"]:
                                st.success("Correct ✅")
                            else:
                                st.error(f"Incorrect ❌. Correct answer: {q['answer']}")

                    practice = generate_practice_offline(
                        row["title"], row["topic"], row["concept_tags"], row["level"]
                    )
                    st.markdown("**Practice Question:**")
                    st.write(practice)
elif mode == "My Courses":
    st.subheader("📚 My Micro-Courses")

    user_progress = progress[
        (progress["user_id"] == user_id) & (progress["watched"] == 1)
    ]

    if user_progress.empty:
        st.info("You have not watched any reels yet. Start from the Learn tab.")
    else:
        course_ids = user_progress["course_id"].dropna().unique().tolist()
        my_courses = courses[courses["course_id"].isin(course_ids)]

        if my_courses.empty:
            st.info("You have watched reels that are not linked to any micro-course.")
        else:
            for _, c in my_courses.iterrows():
                course_videos = videos[videos["course_id"] == c["course_id"]]
                watched_vids = user_progress[user_progress["course_id"] == c["course_id"]]
                total = len(course_videos)
                done = len(watched_vids["video_id"].unique())
                progress_value = done / total if total > 0 else 0

                st.markdown("---")
                st.markdown(f"### {c['course_name']}")
                st.caption(f"Topic: {c['topic']} | Level: {c['level']}")
                st.progress(progress_value)

                if st.button("View Videos", key=f"view_{c['course_id']}"):
                    for _, v in course_videos.iterrows():
                        watched_flag = "✅" if v["video_id"] in watched_vids["video_id"].values else "⭕"
                        st.markdown(
                            f"- {watched_flag} {v['title']} "
                            f"({v['duration_sec']} sec)"
                        )

elif mode == "Creator Studio":
    st.subheader("🎥 Creator Studio – Build Micro-Courses with Reels")

    st.markdown("### Step 1: Create or Select a Micro-Course")

    existing_course_names = ["New Course"] + courses["course_name"].tolist()
    choice = st.selectbox("Choose Course", existing_course_names)

    if choice == "New Course":
        st.markdown("#### Create New Micro-Course")
        new_course_name = st.text_input("Course Name")
        new_course_topic = st.text_input("Course Topic (e.g., DSA, ML, Finance)")
        new_course_level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
        new_course_desc = st.text_area("Course Description")

        if st.button("Create Course"):
            if new_course_name.strip():
                course_id = f"course_{len(courses) + 1}"
                new_row = {
                    "course_id": course_id,
                    "course_name": new_course_name.strip(),
                    "description": new_course_desc.strip(),
                    "level": new_course_level,
                    "topic": new_course_topic.strip(),
                }
                courses = pd.concat([courses, pd.DataFrame([new_row])], ignore_index=True)
                save_data(videos, courses, progress, comments, polls, poll_votes)
                st.success("Micro-course created. Select it from the dropdown above.")
            else:
                st.error("Course name is required.")

    else:
        course_row = courses[courses["course_name"] == choice].iloc[0]
        st.markdown(f"**Selected Course ID:** `{course_row['course_id']}`")
        st.caption(
            f"Topic: {course_row['topic']} | Level: {course_row['level']}\n\n"
            f"{course_row['description']}"
        )

        st.markdown("---")
        st.markdown("### Step 2: Upload a Short Educational Reel (30–90 sec)")

        title = st.text_input("Reel Title")
        topic = st.text_input("Topic", value=str(course_row["topic"]))
        concept_tags = st.text_input(
            "Concept Tags (comma-separated)",
            placeholder="arrays, binary search, time complexity"
        )

        levels_list = ["Beginner", "Intermediate", "Advanced"]
        if course_row["level"] in levels_list:
            default_index = levels_list.index(course_row["level"])
        else:
            default_index = 0

        level = st.selectbox(
            "Skill Level",
            levels_list,
            index=default_index,
            key="reel_skill_level"
        )

        url = st.text_input("Reel URL (YouTube / Shorts / local file path)")
        duration = st.number_input(
            "Duration (seconds, 30–90)",
            min_value=30,
            max_value=90,
            value=60,
            step=5
        )

        st.markdown("### Optional: Add a Quick Poll for Engagement")
        add_poll = st.checkbox("Add a poll to this reel?")
        poll_question, poll_opt1, poll_opt2, poll_opt3, poll_opt4 = None, None, None, None, None

        if add_poll:
            poll_question = st.text_input("Poll Question", placeholder="Was this concept clear?")
            poll_opt1 = st.text_input("Option 1", value="Yes")
            poll_opt2 = st.text_input("Option 2", value="No")
            poll_opt3 = st.text_input("Option 3 (optional)", value="")
            poll_opt4 = st.text_input("Option 4 (optional)", value="")

        if st.button("Upload Reel"):
            if not title.strip() or not url.strip():
                st.error("Title and URL are required.")
            else:
                vid_id = f"video_{len(videos) + 1}"
                new_vid = {
                    "video_id": vid_id,
                    "title": title.strip(),
                    "topic": topic.strip(),
                    "concept_tags": concept_tags.strip(),
                    "level": level,
                    "course_id": course_row["course_id"],
                    "url": url.strip(),
                    "duration_sec": int(duration),
                }

                if "concept_tags" not in videos.columns:
                    videos["concept_tags"] = ""

                videos = pd.concat([videos, pd.DataFrame([new_vid])], ignore_index=True)

                if add_poll and poll_question and (poll_opt1 or poll_opt2):
                    poll_id = f"poll_{len(polls) + 1}"
                    poll_row = {
                        "poll_id": poll_id,
                        "video_id": vid_id,
                        "question": poll_question.strip(),
                        "option_1": poll_opt1.strip(),
                        "option_2": poll_opt2.strip(),
                        "option_3": poll_opt3.strip() if poll_opt3 else "",
                        "option_4": poll_opt4.strip() if poll_opt4 else "",
                    }
                    polls = pd.concat([polls, pd.DataFrame([poll_row])], ignore_index=True)

                save_data(videos, courses, progress, comments, polls, poll_votes)
                st.success("Reel uploaded to micro-course!")

        st.markdown("---")
        st.markdown("### Existing Reels in This Micro-Course")
        course_videos = videos[videos["course_id"] == course_row["course_id"]]
        if course_videos.empty:
            st.caption("No reels in this course yet.")
        else:
            for _, v in course_videos.iterrows():
                st.markdown(
                    f"- {v['title']} ({v['duration_sec']} sec) | "
                    f"Topic: {v['topic']} | Level: {v['level']}"
                )
