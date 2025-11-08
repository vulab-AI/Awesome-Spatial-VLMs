import os

import base64
import requests
from io import BytesIO

import argparse, json, datetime, time, asyncio
import aiohttp
import ssl
import certifi


# Need your API key here
api_key = ""

COMMON_KWARGS = {"temperature": 0.0, "max_tokens": 64}

def genquery(question, answer, ground_truth, prompt):
    return prompt + f"""
Question: {question}
Standard answer: {ground_truth}
Model's answer: {answer}
"""

async def call_chatgpt_async(session, key, question, answer, ground_truth, prompt):

    query = genquery(question, answer, ground_truth, prompt)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": query
            },
        ]
        }
    ],
    "max_tokens": COMMON_KWARGS["max_tokens"],
    "temperature": COMMON_KWARGS["temperature"]
    }

    try:
        success = False
        for _ in range(2):
            try:
                async with session.post(
                        url='https://api.openai.com/v1/chat/completions',
                        headers=headers,
                        json=payload,
                        ssl=ssl.create_default_context(cafile=certifi.where())
                    ) as response:
                    response = await response.json()
                    if "error" in response:
                        # print(f"{key} OpenAI request failed with error {response['error']}")
                        raise Exception(f"OpenAI request failed with error {response['error']}")
                result = response['choices'][0]['message']['content']
                success = True
                break
            except Exception as e:
                print(key, "inner error", e)
                if "content_policy_violation" in e:
                    break
                time.sleep(5)
        if not success:
            raise Exception("Failed to generate")
    except Exception as e:
        result = ""
    return (key, result)

async def call_chatgpt_bulk(keys, questions, answers, ground_truths, prompt):
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as tg:
        responses = [tg.create_task(call_chatgpt_async(
            session, key, question, answer, ground_truth, prompt
        )) for key, question, answer, ground_truth in zip(keys, questions, answers, ground_truths)]
    return responses

def bulk_evaluate(data, batch_size, prompt_item):

    output = []
    for st in range(0, len(data), batch_size):
        ed = st + batch_size
        ed = min(ed, len(data))

        keys = [item["index"] for item in data[st:ed]]
        questions = [item["question"] for item in data[st:ed]]
        answers = [item["answer"] for item in data[st:ed]]
        ground_truths = [item["ground_truth"] for item in data[st:ed]]
        responses = asyncio.run(call_chatgpt_bulk(keys, questions, answers, ground_truths, prompt_item))
        time.sleep(1)

        responses = [response.result() for response in responses]
        output.extend(responses)

    return output