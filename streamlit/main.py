import streamlit as st
import requests
import time

st.title("Essay Feedback and Grading App")

essay = st.text_area("Paste your essay here:", height=300)

if st.button("Get Feedback and Grade"):
    if not essay.strip():
        st.warning("Please enter an essay before submitting.")
    else:
        with st.spinner("Sending essay for analysis..."):
            try:
                post_response = requests.post(
                    "http://localhost:8000/evaluate",
                    json={"text": essay}
                )
                post_response.raise_for_status()
                task_ids = post_response.json()
                qwk_task_id = task_ids["qwk_task_id"]
                feedback_task_id = task_ids["feedback_task_id"]
            except Exception as e:
                st.error(f"Failed to start task: {e}")
                st.stop()

        # Poll for results
        def poll_result(task_id):
            url = f"http://localhost:8000/result/{task_id}"
            for _ in range(30):  # Try for ~30s
                try:
                    res = requests.get(url)
                    res.raise_for_status()
                    result_data = res.json()
                    if result_data.get("status") == "SUCCESS":
                        return result_data.get("result")
                except Exception as e:
                    st.error(f"Error checking task: {e}")
                    break
                time.sleep(1)
            return None

        with st.spinner("Waiting for feedback..."):
            feedback = poll_result(feedback_task_id)

        with st.spinner("Waiting for grade..."):
            qwk = poll_result(qwk_task_id)

        if feedback is not None and qwk is not None:
            st.subheader("📝 Feedback:")
            st.write(feedback)

            st.subheader("🎯 Grade:")
            st.write(qwk)
        else:
            st.error("Failed to get complete results within the timeout.")