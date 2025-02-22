import os
import sys
import json
import fnmatch
import requests
import openai
from unidiff import PatchSet
from zhipuai import ZhipuAI
import re
import gitlab
import hashlib

# 从环境变量中读取必要的参数
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
if not GITLAB_TOKEN:
    raise ValueError("GITLAB_TOKEN is not set")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
if not OPENAI_API_MODEL:
    raise ValueError("OPENAI_API_MODEL is not set")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")
if not OPENAI_API_URL:
    raise ValueError("OPENAI_API_URL is not set")

# GitLab API 的基本地址（默认指向 gitlab.com，如为私有部署请设置 CI_API_V4_URL 环境变量）
CI_API_V4_URL = os.getenv("CI_API_V4_URL", "https://gitlab.com/api/v4")
CI_API = os.getenv("CI_API", "https://gitlab.com")
client = openai.OpenAI(
    base_url=OPENAI_API_URL,
    api_key=OPENAI_API_KEY
)
gl = gitlab.Gitlab(CI_API, private_token=GITLAB_TOKEN)

# client = ZhipuAI(api_key=xxx)
# openai.api_key = OPENAI_API_KEY


#############################################
# 用于构造解析 diff 的数据结构
#############################################
class DiffChange:
    def __init__(self, ln, ln2, content):
        self.ln = ln         # 新增行号（如果存在）
        self.ln2 = ln2       # 原始行号（当新增行号不存在时使用）
        self.content = content

class DiffChunk:
    def __init__(self, content, changes):
        self.content = content  # 包含 hunk 头和具体代码行
        self.changes = changes  # 列表，每一项为 DiffChange 对象

class DiffFile:
    def __init__(self, to, chunks):
        self.to = to         # 目标文件路径
        self.chunks = chunks # 当前文件中的所有代码块

def parse_diff(diff_text):
    """
    利用 unidiff 库将 diff 字符串解析为 DiffFile 列表
    """
    patch = PatchSet(diff_text.splitlines(keepends=True))
    diff_files = []
    for patched_file in patch:
        target = patched_file.target_file
        chunks = []
        for hunk in patched_file:
            header = hunk.section_header.strip() if hunk.section_header else ""
            lines_text = "".join([line.value for line in hunk])
            chunk_content = header + "\n" + lines_text
            changes = []
            for line in hunk:
                # 使用新增行号（target_line_no）为主，否则使用原始行号（source_line_no）
                ln = line.target_line_no
                ln2 = line.source_line_no
                changes.append(DiffChange(ln, ln2, line.value.rstrip("\n")))
            chunks.append(DiffChunk(chunk_content, changes))
        diff_files.append(DiffFile(target, chunks))
    return diff_files


#############################################
# GitLab API 相关函数
#############################################
def get_pr_details(project_id=None, mr_iid=None, ):
    """
    从环境变量中获取项目 ID 和 Merge Request IID，
    并通过 GitLab API 获取 MR 的标题、描述以及 diff refs（用于评论定位）
    """

    if not project_id or not mr_iid:
        raise ValueError("CI_PROJECT_ID and CI_MERGE_REQUEST_IID must be set")
    project = gl.projects.get(project_id)
    mr = project.mergerequests.get(mr_iid)
    diff_refs = mr.diff_refs

    return {
        "project_id": project_id,
        "mr_iid": mr_iid,
        "title": mr.title,
        "description": mr.description,
        "base_sha": diff_refs.get("base_sha"),
        "start_sha": diff_refs.get("start_sha"),
        "head_sha": diff_refs.get("head_sha")
    }


def get_diff(project_id, mr_iid):
    """
    获取 Merge Request 的 diff，调用 .diff 接口返回 raw diff 文本
    """
    project = gl.projects.get(project_id)
    data = project.mergerequests.get(mr_iid).changes()

    unified_diff = ""
    for change in data["changes"]:
        diff = change["diff"]
        old_path = change["old_path"]
        new_path = change["new_path"]
        # 如果 diff 直接以 hunk 开头，则添加必要的文件头信息
        if diff.lstrip().startswith('@@'):
            header = (
                f"diff --git {old_path} {new_path}\n"
                f"--- {old_path}\n"
                f"+++ {new_path}\n"
            )
            diff = header + diff
        unified_diff += diff + "\n"
    return unified_diff if unified_diff.strip() else None


#############################################
# 调用 OpenAI 接口及生成 review 评论相关函数
#############################################
def create_prompt(file, chunk, pr_details):
    """
    根据文件、代码块和 MR 详情构造给 OpenAI 的提示字符串
    """
    diff_changes = ''
    for c in chunk.changes:
        action = 'empty line'
        if c.ln is not None and c.ln2 is not None and c.ln ==c.ln2 :
            action = 'no_change'
        elif c.ln is not None and c.ln2 is not None and c.ln !=c.ln2:
            action = 'Modify'
        elif c.ln is not None :
            action = 'Add'
        elif c.ln2 is not None:
            action = 'Delete'
        change_info = f"old_line:{c.ln2 if c.ln2 is not None else 0}, new_line:{c.ln if c.ln is not None else 0}, action:{action}, content:{c.content}"
        diff_changes = diff_changes + "\n" + change_info

    prompt = f"""Your task is to review merge requests,and reply in chinese. Instructions:
- You Must Provide the response in following JSON format:  {{"reviews": [{{"new_line":  <new_line>, "old_line": <old_line>, "ation": <action>, "reviewComment": "<review comment>"}}]}}
- Do not give positive comments or compliments.
- new_line ,old_line and action in response should exactly the same as the provided diff_change info.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitLab Markdown format.
- Use the given description only for the overall context and only comment the code.
- IMPORTANT: NEVER suggest adding comments to the code.

Review the following code diff in the file "{file.to}" and take the merge request title and description into account when writing the response.

Merge Request title: {pr_details['title']}
Merge Request description:

---
{pr_details['description']}
---

Git diff to review:

```diff
{chunk.content}
{diff_changes}
```"""
    return prompt

def get_ai_response(prompt):
    """
    调用 OpenAI 接口生成代码审查建议，返回一个 reviews 数组，
    每一项格式形如 { "lineNumber": "<line_number>", "reviewComment": "<review comment>" }
    """
    query_config = {
        "model": OPENAI_API_MODEL,
        # "temperature": 0.2,
        # "max_tokens": 700,
        # "top_p": 1,
        # "frequency_penalty": 0,
        # "presence_penalty": 0,
    }
    # 如果模型支持返回 JSON 对象，可传入相应参数
    if OPENAI_API_MODEL == "gpt-4-1106-preview":
        query_config["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(
            **query_config,
            messages=[{"role": "user", "content": prompt}]
        )

        res = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "{}"
        match = re.search(r"\{.*\}", res, re.DOTALL)
        json_str = match.group(0)  # 提取匹配到的 JSON
        print("DEBUG: Extracted JSON:", json_str)
        return json.loads(json_str).get("reviews")
    except Exception as e:
        print("Error from OpenAI:", e)
        return None

def create_comment(file, chunk, ai_responses):
    """
    根据 OpenAI 返回的建议，生成符合 GitLab inline comment 格式的评论列表
    """
    comments = []
    for ai_response in ai_responses:
        if not file.to:
            continue
        comments.append({
            "body": ai_response.get("reviewComment") + '\n ---this is generate by ai!',
            "path": file.to,
            "new_line": int( ai_response.get("new_line")) if ai_response.get("new_line") is not None else 0,
            "old_line": int(ai_response.get("old_line")) if ai_response.get("old_line") is not None else 0,
            "action":ai_response.get("action")
        })
    return comments

def analyze_code(parsed_diff, pr_details):
    """
    遍历所有文件和代码块，调用 OpenAI 获取审查建议，并汇总所有评论
    """
    comments = []
    for file in parsed_diff:
        if file.to == "/dev/null":
            continue  # 忽略已删除的文件
        for chunk in file.chunks:
            prompt = create_prompt(file, chunk, pr_details)
            ai_response = get_ai_response(prompt)
            if ai_response:
                new_comments = create_comment(file, chunk, ai_response)
                if new_comments:
                    comments.extend(new_comments)
        if comments:
            create_review_comments(pr_details, comments)
    return comments


#############################################
# GitLab MR inline 评论相关函数
#############################################
def create_discussion(project_id, mr_iid, comment, pr_details):
    """
    通过 GitLab API 将单条评论以讨论的方式添加到 Merge Request 中
    需要提供 position 信息（基于 MR diff refs）
    """

    project = gl.projects.get(project_id)
    mr = project.mergerequests.get(mr_iid)

    # 构造位置信息（关键参数）
    old_line = comment['old_line']
    new_line = comment['new_line']

    if comment["action"] == 'Add':
        old_line = new_line - 1
    elif comment["action"] == 'Delete':
        new_line = old_line

    position = {
        "position_type": "text",  # 固定值
        "base_sha": mr.diff_refs["base_sha"],
        "head_sha": mr.diff_refs["head_sha"],
        "start_sha": mr.diff_refs["start_sha"],
        "new_path": comment["path"],
        "old_path": comment["path"],

    }
    # 新增/删除 不需要带line_range，修改才需要
    if comment["action"] == "Add":
        position["new_line"] = new_line
    elif comment["action"] == "Delete":
        position["old_line"] = old_line
    else:
        position["new_line"] = new_line
        position["old_line"] = old_line
        position["line_range"]: {
            "start": {
                "line_code": generate_line_code(comment["path"], old_line,new_line),  # line_code计算规则
                "old_line":  old_line,
                "new_line":   new_line
            },
            "end": {
                 "line_code": generate_line_code(comment["path"], old_line,new_line),  # line_code计算规则
                 "old_line":  old_line,
                 "new_line":   new_line
            }
        }

    # 创建行内评论
    discussion = mr.discussions.create({
        "body": comment["body"],
        "position": position
    })
    print(discussion)

def generate_line_code(fileName, old_line=None, new_line=None):
    """ 生成 GitLab `line_code`（唯一标识某一行） """
    return f"{hashlib.sha1(fileName.encode()).hexdigest()}_{old_line or 0}_{new_line or 0}"

def create_review_comments(pr_details, comments):
    """
    将所有评论逐条以讨论的方式发布到 Merge Request 中
    """
    project_id = pr_details["project_id"]
    mr_iid = pr_details["mr_iid"]
    for comment in comments:
        create_discussion(project_id, mr_iid, comment, pr_details)


#############################################
# 主函数
#############################################
def start_ai_code_review(project_name=None, project_id=None, merge_id=None, branch=None, target_branch=None):
    try:
        # 获取 MR 详情（标题、描述、diff refs 等）
        pr_details = get_pr_details(project_id, merge_id)
        project_id = pr_details["project_id"]
        mr_iid = pr_details["mr_iid"]

        # 获取 Merge Request 的 diff（raw diff 格式）
        diff = get_diff(project_id, mr_iid)
        if not diff:
            print("No diff found")
            return

        # 解析 diff，构造文件、代码块等数据结构
        parsed_diff = parse_diff(diff)

        # 根据环境变量 INPUT_EXCLUDE 排除不需要处理的文件（逗号分隔）
        exclude_input = os.getenv("INPUT_EXCLUDE", "vendor/**,test/**")
        exclude_patterns = [s.strip() for s in exclude_input.split(",") if s.strip()]
        filtered_diff = [
            file for file in parsed_diff
            if not any(fnmatch.fnmatch(file.to, pattern) for pattern in exclude_patterns)
        ]

        # 调用 OpenAI 分析代码 diff，生成 review 评论
        comments = analyze_code(filtered_diff, pr_details)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    project = ""
    source_branch = ""
    target_branch = ""
    project_id = ""
    merge_id = ""
    try:
        project = sys.argv[1]
        source_branch = sys.argv[2]
        target_branch = sys.argv[3]
        project_id = sys.argv[4]
        merge_id = sys.argv[5]
    except Exception as e:
        print(e)
        sys.exit(1)
    start_ai_code_review(project,project_id,merge_id,source_branch,target_branch)

