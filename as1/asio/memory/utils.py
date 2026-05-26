import re
from typing import Any, Sequence, Union
from agentscope.message import Msg
import json
from transformers import AutoTokenizer
import copy
import traceback

def output_format(item: Msg) -> Any:
    copy_item = copy.deepcopy(item)
    processed_item = {}
    if hasattr(copy_item, 'role'):
        processed_item['role'] = copy_item.role
    
    if hasattr(copy_item, 'content'):
        data = copy_item.content
        
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                if not isinstance(parsed_data, dict):
                    processed_item['data'] = [s for s in re.split(r'\n+', data) if s] if '\n' in data else data
                else:
                    processed_item['data'] = data
            except json.JSONDecodeError:
                processed_item['data'] = [s for s in re.split(r'\n+', data) if s] if '\n' in data else data
        else:
            processed_item['data'] = data
    if 'data' in processed_item and isinstance(processed_item['data'], list) and processed_item['data'] and isinstance(processed_item['data'][0], dict):
        for dict_item in processed_item['data']:
            output = dict_item.get('output', None)
            if output is None:
                continue
            if isinstance(output, str):
                dict_item['output'] = [s for s in re.split(r'\n+', output) if s]
            elif isinstance(output, list) and output and isinstance(output[0], dict):
                for out_item in output:
                    for key, value in out_item.items():
                        if isinstance(value, str):
                            out_item[key] = [s for s in re.split(r'\n+', value) if s] if '\n' in value else value
    return processed_item

def time_order_check(unit: Sequence[Any]) -> bool:
    """Check if the unit is in time order."""

    def get_time(unit: Any) -> str:
        try:
            return unit.payload.get("last_modified_timestamp", None)
        except Exception:
            raise ValueError(f"Invalid unit type: {type(unit)}")

    for i in range(len(unit) - 1):
        time_i, time_i_1 = get_time(unit[i]), get_time(unit[i + 1])
        if time_i is not None and time_i_1 is not None and time_i > time_i_1:
            return False
    return True


def format_msgs(
    msgs: Union[Sequence[Msg], Msg],
    with_id: bool = True,
    strip_content_ids: bool = False,
) -> str:
    """Format a list of messages to a string in order.

    Args:
        msgs (Union[Sequence[Msg], Msg]):
            The messages to format.
        with_id (bool):
            Whether to include message IDs in the output.
        strip_content_ids (bool):
            Whether to strip 'id' fields from tool_use/tool_result content items.
            Useful when building prompts where internal IDs add noise.

    Raises:
        ValueError: The message type or the content type is invalid.

    Returns:
        str: The formatted messages as a JSON string.
    """
    results = []
    if not isinstance(msgs, Sequence):
        msgs = [msgs]
    for idx, msg in enumerate(msgs):
        if not isinstance(msg, Msg):
            raise ValueError(f"Invalid message type: {type(msg)}")

        role = msg.role
        content = msg.content
        if content is None:
            content = ""
        if isinstance(content, str):
            results.append(
                {
                    "role": role,
                    "content": content,
                },
            )
            if with_id:
                results[-1]["id"] = idx
            continue
        elif isinstance(content, list):
            unit = {
                "role": role,
                "content": [],
            }
            if with_id:
                unit["id"] = idx
            for c in content:
                if strip_content_ids and isinstance(c, dict) and c.get("type") in ("tool_use", "tool_result") and "id" in c:
                    c = {k: v for k, v in c.items() if k != "id"}
                unit["content"].append(c)
            results.append(unit)
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
    return json.dumps(results, ensure_ascii=False)


def count_words(tokenizer: AutoTokenizer, text: Any) -> int:
    """Count the number of tokens in the text.

    Args:
        text (Any):
            the text to count the number of tokens

    Returns:
        int: the number of tokens in the text
    """
    if not isinstance(text, str):
        text = str(text)
    return len(tokenizer.encode(text))

def _get_file_block_encoded(file_list: list[str]) -> str:
    if file_list:
        return json.dumps(file_list)
    else:
        return ""

def _get_file_block_decoded(file_block: str) -> list[str]:
    if file_block:
        return json.loads(file_block)
    else:
        return []

def split_text(tokenizer: Any, content: str, token_limit: int, overlap: int=0):
    tokens = tokenizer.encode(content, add_special_tokens=False)
    tot_len = len(tokens)
    content_list = []
    if tot_len < token_limit:
        content_list.append(content)
    else:
        added_len = 0
        while added_len < tot_len:
            split_token = tokens[:token_limit]
            truncated_text = tokenizer.decode(split_token)
            content_list.append(truncated_text)
            tokens = tokens[token_limit-overlap:]
            added_len += token_limit-overlap
    return content_list 

def has_tool_result(msgs: Sequence[Msg]):
    has_tool_result = False
    for msg in msgs:
        if isinstance(msg.content, list) and (any(
            c.get("type", None) == "tool_result" for c in msg.content) or any(
            c.get("type", None) == "text" and c.get("text", "").startswith("<tool_result>") for c in msg.content)):
            has_tool_result = True
            break
    return has_tool_result

def has_tool_use(msgs: Sequence[Msg]):
    has_tool_use = False
    for msg in msgs:
        if isinstance(msg.content, list) and (any(
            c.get("type", None) == "tool_use" for c in msg.content) or any(
            c.get("type", None) == "text" and c.get("text", "").startswith("<tool_call>") for c in msg.content)):
            has_tool_use = True
            break
    return has_tool_use

def try_json_loads(s: str):
    try:
        return json.loads(s), None
    except json.JSONDecodeError as e:
        return None, e

def repair_unescaped_inner_quotes(s: str) -> str:
    out = []
    in_string = False
    escape = False  # Whether the previous character is a backslash that hasn't been consumed yet

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]

        if not in_string:
            # Not in a string, only enter string when encountering an unescaped "
            if ch == '"':
                in_string = True
                escape = False
                out.append(ch)
            else:
                out.append(ch)
            i += 1
            continue

        # Inside string
        if escape:
            # Current character is escaped (including \", \\, \n, \u, etc.), keep as is
            out.append(ch)
            escape = False
            i += 1
            continue

        if ch == '\\':
            # Enter escape state, next character will be escaped
            out.append(ch)
            escape = True
            i += 1
            continue

        if ch == '"':
            # Unescaped double quote, determine if it should be treated as a closing quote
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            next_ch = s[j] if j < n else None
            is_closing = next_ch in (',', '}', ']', None)
            if is_closing:
                # Closing quote
                out.append(ch)
                in_string = False
            else:
                # Inner quote: insert a backslash before it to escape
                out.append('\\')
                out.append('"')
            i += 1
            continue

        # Regular character
        out.append(ch)
        i += 1

    return ''.join(out)

def tolerant_loads(s: str, special_json_patterns: list[str] = []):
    obj, err = try_json_loads(s)
    if obj is not None:
        return obj
    
    # Remove markdown code block markers
    s = re.sub(r'```json\s*|\s*```', '', s).strip()
    obj, err = try_json_loads(s)
    if obj is not None:
        return obj
    
    # Try to extract JSON from mixed text content
    # Look for JSON object patterns
    json_patterns = [
        r'\{.*?\}',  # General JSON object pattern
        r'\[.*?\]',  # JSON array pattern
    ].extend(special_json_patterns)
    
    for pattern in json_patterns:
        matches = re.findall(pattern, s, re.DOTALL)
        for match in matches:
            obj, _ = try_json_loads(match.strip())
            if obj is not None:
                return obj
    
    # Try to find JSON between braces more aggressively
    brace_count = 0
    start_pos = -1
    for i, char in enumerate(s):
        if char == '{':
            if start_pos == -1:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_pos != -1:
                potential_json = s[start_pos:i+1]
                obj, _ = try_json_loads(potential_json.strip())
                if obj is not None:
                    return obj
                start_pos = -1
    
    # Try repairing unescaped quotes as last resort
    repaired = repair_unescaped_inner_quotes(s)
    obj2, err2 = try_json_loads(repaired)
    if obj2 is not None:
        return obj2

    # If still failed, throw friendly error with traceback
    tb = traceback.format_exc()
    raise ValueError(f"JSON parsing failed. Original error: {err}; Error after repair: {err2}; Original string: {s[:200]}...\nTraceback: {tb}")


def check_tool_use_result_paired(msgs: Sequence[Msg]) -> bool:
    tool_use_ids = set()
    tool_result_ids = set()

    for msg in msgs:
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict):
                    if item.get("type") == "tool_use" and "id" in item:
                        tool_use_ids.add(item["id"])
                    elif item.get("type") == "tool_result" and "id" in item:
                        tool_result_ids.add(item["id"])

    return tool_use_ids == tool_result_ids


def remove_thinking_block(s: str) -> str:
    return re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)