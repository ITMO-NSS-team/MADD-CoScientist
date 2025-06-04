import streamlit as st
import time
from streamlit_endpoints import get_answer_from_assistant, process_uploaded_file

# Page configuration
st.set_page_config(
    page_title="Chemical Assistent",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Main title
st.title("💬 Chemical Assistent")

# Sidebar for file uploads and chat management
with st.sidebar:
    st.header("📁 File Upload")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload documents to add to knowledge base",
        type=['pdf'],
        help="Supported formats: PDF"
    )

    if uploaded_file is not None:
        if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            with st.spinner("Processing file..."):
                try:
                    # Process the uploaded file
                    result = process_uploaded_file(uploaded_file)

                    if result["success"]:
                        st.session_state.uploaded_files.append({
                            "name": uploaded_file.name,
                            "size": uploaded_file.size,
                            "type": uploaded_file.type
                        })
                        st.write(result['msg'])
                        st.success(f"✅ {result['msg']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Error processing file: {result['msg']}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("📚 Uploaded Files")
        for file_info in st.session_state.uploaded_files:
            with st.expander(f"📄 {file_info['name']}", expanded=False):
                st.write(f"**Size:** {file_info['size']:,} bytes")
                st.write(f"**Type:** {file_info['type']}")

    st.divider()

    # Chat management
    st.header("🔧 Chat Management")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        time.sleep(1)
        st.rerun()

    # Display chat statistics
    if st.session_state.messages:
        st.metric("Messages", len(st.session_state.messages))
        user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        ai_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
        st.metric("User Messages", user_messages)
        st.metric("AI Responses", ai_messages)

# Main chat interface
st.header("💭 Chat")

# Display chat messages
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info("👋 Welcome! Start a conversation by typing a message below.")
    else:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"*{message['timestamp']}*")

                # Show context buttons for assistant messages that have additional data
                if (message["role"] == "assistant" and
                        any(key in message for key in ["text_context", "images_context", "metadata"])):

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        show_hist_text = st.checkbox("📄 Text Context", key=f"hist_text_{i}")

                    with col2:
                        show_hist_images = st.checkbox("🖼️ Image Context", key=f"hist_image_{i}")

                    with col3:
                        show_hist_meta = st.checkbox("ℹ️ Metadata", key=f"hist_meta_{i}")

                    # Display contexts if selected
                    if show_hist_text:
                        with st.expander("📄 Text Context", expanded=True):
                            st.text_area("Text Context:", value=message.get("text_context", ""), height=200,
                                         disabled=True, key=f"text_area_{i}")

                    if show_hist_images:
                        with st.expander("🖼️ Image Context", expanded=True):
                            images_context = message.get("images_context", [])
                            if images_context:
                                for j, image_item in enumerate(images_context):
                                    show_hist_img = st.checkbox(f"{j + 1}. {image_item}",
                                                                key=f"hist_img_checkbox_{i}_{j}")

                                    # Display image if selected
                                    if show_hist_img:
                                        try:
                                            st.image(image_item, caption=image_item, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Could not display image: {image_item}. Error: {str(e)}")
                            else:
                                st.write("No image context available")

                    if show_hist_meta:
                        with st.expander("ℹ️ Metadata", expanded=True):
                            metadata = message.get("metadata", {})
                            if metadata:
                                for key, value in metadata.items():
                                    st.write(f"**{key}:** {value}")
                            else:
                                st.write("No metadata available")

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    }
    st.session_state.messages.append(user_message)

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(f"*{timestamp}*")

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call the AI response function
                text_context, images_context, answer, metadata = get_answer_from_assistant(user_input)

                # Display the answer
                st.markdown(answer)

                # Create toggles for context display
                col1, col2, col3 = st.columns(3)

                msg_idx = len(st.session_state.messages)

                with col1:
                    show_text = st.checkbox("📄 Text Context", key=f"text_context_{msg_idx}")

                with col2:
                    show_images = st.checkbox("🖼️ Image Context", key=f"image_context_{msg_idx}")

                with col3:
                    show_meta = st.checkbox("ℹ️ Metadata", key=f"metadata_{msg_idx}")

                # Display text context if selected
                if show_text:
                    with st.expander("📄 Text Context", expanded=True):
                        st.text_area("Text Context:", value=text_context, height=200, disabled=True)

                # Display image context if selected
                if show_images:
                    with st.expander("🖼️ Image Context", expanded=True):
                        if images_context:
                            for i, image_item in enumerate(images_context):
                                show_img = st.checkbox(f"{i + 1}. {image_item}", key=f"img_checkbox_{msg_idx}_{i}")

                                # Display image if selected
                                if show_img:
                                    try:
                                        st.image(image_item, caption=image_item, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not display image: {image_item}. Error: {str(e)}")
                        else:
                            st.write("No image context available")

                # Display metadata if selected
                if show_meta:
                    with st.expander("ℹ️ Metadata", expanded=True):
                        if metadata:
                            for key, value in metadata.items():
                                st.write(f"**{key}:** {value}")
                        else:
                            st.write("No metadata available")

                # Add AI response to chat history
                ai_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                ai_message = {
                    "role": "assistant",
                    "content": answer,
                    "text_context": text_context,
                    "images_context": images_context,
                    "metadata": metadata,
                    "timestamp": ai_timestamp
                }
                st.session_state.messages.append(ai_message)
                st.caption(f"*{ai_timestamp}*")

            except Exception as e:
                error_message = f"❌ Sorry, I encountered an error: {str(e)}"
                st.error(error_message)

                # Add error message to chat history
                error_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                error_msg = {
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": error_timestamp
                }
                st.session_state.messages.append(error_msg)

# Footer information
st.divider()
st.caption("💡 **Tip:** Upload documents to enhance the AI's knowledge base and get more accurate responses!")

# Display system status
with st.expander("ℹ️ System Information", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Chat Messages", len(st.session_state.messages))

    with col2:
        st.metric("Uploaded Files", len(st.session_state.uploaded_files))

    with col3:
        total_file_size = sum(file["size"] for file in st.session_state.uploaded_files)
        st.metric("Total File Size", f"{total_file_size:,} bytes")
