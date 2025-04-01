from openai import AzureOpenAI, OpenAIError, RateLimitError, APIError
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

from django.utils import timezone
from django.conf import settings

from enum import Enum
from typing import Optional
from asgiref.sync import sync_to_async
from bs4 import BeautifulSoup
from datetime import timedelta

import tiktoken
import asyncio
import re
import markdown
import os
import time
import html

from .prompt_temps import grammar_prompt, email_prompt, meeting_insights_prompt, document_prompt
from .models import LAChatMessage

import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Azure OpenAI KEY and ENDPOINT must be set in the environment.")


def count_tokens(text, model="gpt-4o"):
    """Estimate token count for a given text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


class TaskType(Enum):
    GRAMMAR_CHECK = "grammar"
    EMAIL_WRITE = "email"
    MEETING_INSIGHTS = "meeting_insights"
    DOCUMENT = "document"
    
    
class LanguageProcessor:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model = model_name
        self.prompt_dict = {
            "grammar": grammar_prompt(),
            "email": email_prompt(),
            "meeting_insights": meeting_insights_prompt(),
            "document": document_prompt()
        }

        # Create an AzureOpenAI client here
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        
        self.max_token = 16384
        self.temperature = 0.1

        # templates logic
        self.templates = {
            bot_type: self.create_prompt_template(bot_type)
            for bot_type in self.prompt_dict
        }


    async def generate_embeddings(self, input_text):
        """Generate embeddings using Azure OpenAI SDK (Faster than REST API)."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",  
                input=[input_text]
            )
            return response.data[0].embedding

        except Exception as e:
            print(f"Embedding API Error: {e}")
            return None
        
    def get_chat_response(self, messages):
        retry_count = 0
        max_retry = 3
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_content_token_count": 0,
            'llm_response_time': 0
        }

        while retry_count < max_retry:
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_token,
                    temperature=self.temperature,
                )
                answer = response.choices[0].message.content
                usage_data = response.usage
                llm_response_time = time.time() - start_time
                usage = {
                    "input_tokens": int(usage_data.prompt_tokens) if usage_data.prompt_tokens is not None else 0,
                    "output_tokens": int(usage_data.completion_tokens) if usage_data.completion_tokens is not None else 0,
                    "cached_content_token_count": 0,
                    'llm_response_time': llm_response_time
                }
                return {
                    'answer': answer,
                    'usage': usage,
                    'error': None
                }
                
            except (RateLimitError, APIError) as e:
                openai_error = str(e)
                logger.error(f"Unexpected error in get_chat_response: {openai_error}, Attemp No: {retry_count}")
                retry_count += 1
                wait_time = 3 * retry_count
                time.sleep(wait_time)
                
            except OpenAIError as e:
                openai_error = str(e)
                logger.error(f"Unexpected error in get_chat_response: {openai_error}")
                return {'error': openai_error, 'usage': usage, 'answer': None}
            
            except Exception as e:
                openai_error = str(e)
                logger.error(f"Unexpected error in get_chat_response: {openai_error}")
                return {'error': openai_error, 'usage': usage, 'answer': None}
            
        return {'error': openai_error, 'usage': usage, 'answer': None}


    def create_prompt_template(self, bot_type) -> ChatPromptTemplate:
        system_template = self.prompt_dict.get(bot_type)
        human_template = """                    
            Question: {question}
            Answer:
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ]
        )
        return prompt_template

    async def process_query(
        self, 
        input_text: str, 
        bot_type: TaskType, 
        session_id: Optional[str] = None, 
        **kwargs
        ) -> tuple:
        """
        Process text based on specified task type with additional parameters.

        Args:
            input_text (str): The text to process
            bot_type (TaskType): Type of processing required
            session_id (str): Identifier for conversation history (multi-turn)
            **kwargs: Additional parameters

        Returns:
            tuple: (response_text, token_usage_dictionary)
        """
        try:
            task_prompt = self.prompt_dict.get(bot_type)
            if not task_prompt:
                raise ValueError(f"Invalid task type: {bot_type}")
            
            # Fetch chat history
            chat_history = []
            if session_id:
                try:
                    chat_history = await fetch_chat_history(session_id, bot_type)
                except Exception as e:
                    logger.warning(f"Failed to fetch chat history: {str(e)}")

            history_text = ""
            if chat_history:
                for chat in chat_history:
                    if chat.user_message:
                        history_text += f"User: {chat.user_message}\n"
                    if chat.bot_response:
                        history_text += f"Assistant: {chat.bot_response}\n"

            system_prompt = f"""
                Query Instructions: {task_prompt}
                Previous Conversation History : {history_text}
            """

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]
            
            chat_response = self.get_chat_response(messages)
            answer = chat_response.get("answer", "")
            usage = chat_response.get("usage", {})
            error =  chat_response.get("error", "")
                
            # # Post-process to ensure consistent formatting
            # if answer and not bool(re.search(r'<[^>]+>', answer)):
            #     answer = f"<div>\n{answer}\n</div>"

            return {'answer': answer, 'usage': usage, "error": error}
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(error_msg)
            logger.error(f"Error in function 'process_query': {error_msg}")
            return {'answer': answer, 'usage': usage, "error": error_msg}


@sync_to_async
def fetch_chat_history(session_id: str, bot_type: TaskType, pair_limit=5):
    """
    Synchronously fetch the last `pair_limit` user+assistant pairs
    from DB, from an async context.
    Returns them in chronological order (oldest first).
    """
    message_limit = pair_limit

    qs = (
        LAChatMessage.objects
        .filter(session_id=session_id, bot_type=bot_type)
        .order_by('-created_at')[:message_limit]  # get the newest 'message_limit' rows
    )
    # Reverse them so the oldest among this slice is first
    return list(qs[::-1])


def process_text_by_task(
        bot_type,
        input_text,
        login_key=None,
):
    """
    Processes the input text based on the task type.
    Optionally uses multi-turn if session_id is provided.
    Includes retry mechanism with exponential backoff.
    """

    try:
        if not hasattr(process_text_by_task, 'processor'):
            process_text_by_task.processor = LanguageProcessor()

        response = asyncio.run(
            process_text_by_task.processor.process_query(
                input_text, bot_type, session_id=login_key
            )
        )

        answer = response.get("answer", "")
        usage = response.get("usage", {})
        error =  response.get("error", "")

        if usage and isinstance(usage, dict):
            calculate_cost = calculate_usage_cost(usage)
            total_cost = calculate_cost.get('total_cost', 0)
            usage.update({'total_cost': total_cost})
            
        # if answer:
        #     answer = format_answer(answer)
            
        return {'answer': answer, 'usage': usage, "error": error}
    except Exception as error:
        print(error)
        logger.error(f"Error in function 'process_text_by_task': {str(error)}")
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_content_token_count": 0,
            "llm_response_time": 0,
            "total_cost": 0
        }
        return {'answer': None, 'usage': usage, "error": error}


def format_answer(answer):
    """
    Converts HTML to clean Markdown and handles various response formats.
    
    Args:
        answer (str): The input HTML/text content to be formatted
        
    Returns:
        str: Clean Markdown formatted text
    """
    is_html = bool(re.search(r'<[^>]+>', answer))
    
    if is_html:
        soup = BeautifulSoup(answer, 'html.parser')
        
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(tag.name[1])
            tag.replace_with(f"{'#' * level} {tag.get_text().strip()}\n\n")
            
        for ol in soup.find_all('ol'):
            items = ol.find_all('li')
            for i, item in enumerate(items, 1):
                item.replace_with(f"{i}. {item.get_text().strip()}\n")
        
        for ul in soup.find_all('ul'):
            items = ul.find_all('li')
            for item in items:
                item.replace_with(f"- {item.get_text().strip()}\n")
        
        for p in soup.find_all('p'):
            p.replace_with(f"{p.get_text().strip()}\n\n")
            
        for strong in soup.find_all(['strong', 'b']):
            strong.replace_with(f"**{strong.get_text().strip()}**")
            
        for em in soup.find_all(['em', 'i']):
            em.replace_with(f"*{em.get_text().strip()}*")
            
        for br in soup.find_all('br'):
            br.replace_with('\n')
            
        markdown_text = soup.get_text()
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text) 
        markdown_text = re.sub(r'[ \t]+\n', '\n', markdown_text) 
        markdown_text = re.sub(r'\n+$', '\n', markdown_text)  
        
    else:
        markdown_text = answer.strip()
    
    markdown_text = html.unescape(markdown_text)
    markdown_text = re.sub(r'\\([\\`*_{}[\]()#+\-.!])', r'\1', markdown_text)
    markdown_text = re.sub(r'\s*\n\s*\n\s*\n\s*', '\n\n', markdown_text)
    markdown_text = markdown_text.strip()
    
    html_content = markdown.markdown(
        markdown_text,
        extensions=['extra', 'nl2br', 'sane_lists']
    )
    
    html_content = re.sub(r'<hr\s*/?>', '', html_content)
    return html_content.strip()


def verify_session_timeout(request):
    current_time = timezone.now()
    last_activity = request.session.get('last_activity')
    if last_activity:
        last_activity = timezone.datetime.fromisoformat(last_activity)
        if current_time > last_activity + timedelta(seconds=settings.SESSION_COOKIE_AGE):
            return False

    request.session['last_activity'] = current_time.isoformat()
    request.session.modified = True
    return True
    

def calculate_usage_cost(
        usage_token: dict,
        model: str = "gpt-4o",
        pricing_rates: dict = None
) -> dict:
    """
    Calculate the usage cost based on the provided usage token and pricing rates.

    For Azure OpenAI:
    - input tokens: $2.5 / 1M tokens
    - output tokens: $10 / 1M tokens
    """
    try:
        input_token_cost = 2.5
        output_token_cost = 10

        input_tokens = usage_token.get('input_tokens', 0)
        output_tokens = usage_token.get('output_tokens', 0)

        input_cost = (input_tokens / 1000000) * input_token_cost
        output_cost = (output_tokens / 1000000) * output_token_cost
        total_cost = input_cost + output_cost

        return {
            "model": model,
            "breakdown": {
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
            },
            "total_cost": round(total_cost, 4)
        }
    except Exception as e:
        print(f"error in cost module: {e}")
        logger.error(f"Error in function 'calculate_usage_cost': {str(e)}")
        return {
            "model": model,
            "breakdown": {
                "input_cost": 0,
                "output_cost": 0,
            },
            "total_cost": 0
        }


