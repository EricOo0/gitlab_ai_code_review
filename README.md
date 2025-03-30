# AI-Code-Review

# 基本功能
本项目支持接入gitlab cicd pipeline，可以实现自动拉取mr diff，调用LLM进行code review，并自动在对应行进行comment。

# 配置要求
> python3.9
> 
> 执行以下命令安装依赖
```shell    
pip3 install -r requirements
```

# 环境变量配置
> 兼容支持openAI API格式大模型
> 
> 需要设置以下环境变量，用于调用LLM
```shell
export GITLAB_TOKEN = 'YOUR GITLAB_TOKEN'
export OPENAI_API_KEY = 'YOUR OPENAI_API_KEY'
export OPENAI_API_MODEL = 'YOUR OPENAI_API_MODEL'
export OPENAI_API_URL = 'YOUR OPENAI_API_URL'
// 以下域名如果是私有化部署改成对应的url
export CI_API_V4_URL = 'https://gitlab.com/api/v4'
export CI_API = 'https://gitlab.com'

```

# 本地使用
```shell
python3 main.py "" "" "" your_project_id your_mergeid

```
# 接入gitlab cicd pipeline使用
main.py 放到对应项目script文件夹下

gitlab.yml增加一个stage（路径不同需要稍微修改stage.script）

PS. runner上需要先准备好对应的环境(pyenv的python3.9环境)
```shell
review:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  tags:
      - your_runner_tag
  stage: review
  allow_failure: true
  script:
    - cd script
    - ls -al
    - echo $CI_PROJECT_NAME
    - echo $GITLAB_USER_EMAIL
    - echo $CI_PROJECT_NAME
    - echo $CI_COMMIT_REF_NAME
    - echo $CI_MERGE_REQUEST_SOURCE_BRANCH_NAME
    - echo $CI_MERGE_REQUEST_TARGET_BRANCH_NAME
    - echo $CI_MERGE_REQUEST_ASSIGNEES
    - echo $CI_PROJECT_ID
    - whoami
    - source ~/.bashrc
    - echo $PATH
    - pyenv global 3.9
    - pip3 install -r requirements.txt  --break-system-packages
    - python3 main.py $CI_PROJECT_NAME $CI_MERGE_REQUEST_SOURCE_BRANCH_NAME $CI_MERGE_REQUEST_TARGET_BRANCH_NAME $CI_PROJECT_ID $CI_MERGE_REQUEST_IID

```

# OTher
Rag  文件夹下为 rag 操作流的简单demo
简单演示了 查询 -> 查询改写 -> 知识导入&查询 -> 总结 -> 提问的流程

deep researrch 文件夹为 deep research 的流程演示，关键区别在于工具的使用(还未完成)

transaction 目录增加A股选股指标计算&建议Demo，后续尝试将指标提供给LLM进行选股建议
