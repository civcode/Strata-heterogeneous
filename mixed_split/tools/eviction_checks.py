#!/usr/bin/env python3
"""Exercise four independent chats and report each turn's cached-token count."""
import http.client
import json
import sys

port = int(sys.argv[1])
model = sys.argv[2]
chats = {}


def turn(name, text):
    messages = chats.setdefault(name, [])
    messages.append({'role': 'user', 'content': text})
    body = {'model': model, 'messages': messages, 'temperature': 0, 'max_tokens': 80,
            'reasoning_budget_tokens': 0}
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
    conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    if response.status != 200:
        raise RuntimeError(f'{name}: HTTP {response.status}: {payload}')
    answer = payload['choices'][0]['message']['content'] or ''
    messages.append({'role': 'assistant', 'content': answer})
    usage = payload['usage']
    print(json.dumps({'chat': name, 'turn': len(messages) // 2, 'cached':
                      usage.get('prompt_tokens_details', {}).get('cached_tokens'),
                      'prompt_tokens': usage.get('prompt_tokens'), 'answer': answer}), flush=True)


for name, topic in [('A', 'oceans'), ('B', 'volcanoes'), ('C', 'rainforests'), ('D', 'deserts')]:
    turn(name, f'In one sentence, explain one interesting fact about {topic}.')
    turn(name, 'Give one more short fact about the same subject.')
for name in ('B', 'A', 'D'):
    turn(name, 'Give one final short fact about the same subject.')
