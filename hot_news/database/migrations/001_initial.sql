-- =====================================================
-- SentimentSpider - Hot News Module Database Schema
-- =====================================================

-- 热点平台配置表
CREATE TABLE IF NOT EXISTS hot_platform_config (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    platform_code VARCHAR(50) NOT NULL UNIQUE COMMENT '平台代码',
    platform_name VARCHAR(100) NOT NULL COMMENT '平台名称',
    is_enabled TINYINT DEFAULT 1 COMMENT '是否启用：0-禁用，1-启用',
    priority INT DEFAULT 0 COMMENT '优先级，数字越大越优先',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_enabled (is_enabled),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='热点平台配置表';

-- 领域配置表
CREATE TABLE IF NOT EXISTS domain_config (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    domain_name VARCHAR(100) NOT NULL UNIQUE COMMENT '领域名称',
    domain_keywords TEXT COMMENT '领域关键词，多个关键词用逗号分隔',
    is_enabled TINYINT DEFAULT 1 COMMENT '是否启用：0-禁用，1-启用',
    description VARCHAR(500) COMMENT '领域描述说明',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_enabled (is_enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='领域配置表';

-- 大模型配置表
CREATE TABLE IF NOT EXISTS llm_config (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    provider VARCHAR(50) NOT NULL COMMENT 'LLM提供商：deepseek、qwen等',
    api_base VARCHAR(500) COMMENT 'API接口地址',
    api_key VARCHAR(500) COMMENT 'API密钥',
    model_name VARCHAR(100) COMMENT '模型名称',
    temperature DECIMAL(3,2) DEFAULT 0.7 COMMENT '温度参数，控制生成随机性',
    max_tokens INT DEFAULT 2000 COMMENT '最大生成token数',
    is_enabled TINYINT DEFAULT 1 COMMENT '是否启用：0-禁用，1-启用',
    is_default TINYINT DEFAULT 0 COMMENT '是否默认配置：0-否，1-是',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_enabled (is_enabled),
    INDEX idx_default (is_default)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='大模型配置表';

-- 爬虫平台配置表
CREATE TABLE IF NOT EXISTS crawler_platform_config (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    platform_code VARCHAR(50) NOT NULL UNIQUE COMMENT '平台代码',
    platform_name VARCHAR(100) NOT NULL COMMENT '平台名称',
    is_enabled TINYINT DEFAULT 1 COMMENT '是否启用：0-禁用，1-启用',
    max_notes_count INT DEFAULT 50 COMMENT '单次爬取最大笔记/视频数量',
    max_comments_count INT DEFAULT 20 COMMENT '单条内容最大爬取评论数量',
    priority INT DEFAULT 0 COMMENT '优先级，数字越大越优先',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_enabled (is_enabled),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='爬虫平台配置表';

-- 任务调度配置表
CREATE TABLE IF NOT EXISTS task_schedule_config (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    task_name VARCHAR(100) NOT NULL UNIQUE COMMENT '任务名称，唯一标识',
    cron_expression VARCHAR(100) COMMENT 'Cron表达式，任务执行时间规则',
    is_enabled TINYINT DEFAULT 1 COMMENT '是否启用：0-禁用，1-启用',
    max_hot_count INT DEFAULT 100 COMMENT '每次任务最多获取热点数量',
    need_llm_check TINYINT DEFAULT 1 COMMENT '是否需要LLM进行领域匹配检查：0-否，1-是',
    need_keyword_extract TINYINT DEFAULT 1 COMMENT '是否需要LLM提取关键词：0-否，1-是',
    execute_immediately TINYINT DEFAULT 0 COMMENT '是否立即执行：0-否，1-是',
    last_execute_time DATETIME COMMENT '最后执行时间',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_enabled (is_enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务调度配置表';

-- 热点新闻表
CREATE TABLE IF NOT EXISTS hot_news (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    news_id VARCHAR(100) NOT NULL UNIQUE COMMENT '热点新闻唯一ID，由来源平台+标题哈希生成',
    platform_code VARCHAR(50) NOT NULL COMMENT '来源平台代码',
    title VARCHAR(500) NOT NULL COMMENT '热点新闻标题',
    url VARCHAR(1000) COMMENT '原文链接地址',
    score VARCHAR(50) COMMENT '热度评分/热度值',
    description TEXT COMMENT '热点新闻描述/摘要',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
    INDEX idx_platform (platform_code),
    INDEX idx_created (created_at),
    INDEX idx_news_id (news_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='热点新闻表';

-- 热点分析结果表
CREATE TABLE IF NOT EXISTS hot_news_analysis (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    news_id VARCHAR(100) NOT NULL COMMENT '关联hot_news表的news_id',
    domain_id INT COMMENT '关联domain_config表的id',
    is_match TINYINT COMMENT '是否匹配该领域：0-否，1-是',
    llm_provider VARCHAR(50) COMMENT '使用的LLM提供商',
    analysis_content TEXT COMMENT 'LLM分析的具体内容',
    confidence DECIMAL(5,2) COMMENT '置信度，0-1之间的小数',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '分析时间',
    UNIQUE KEY uk_news_domain (news_id, domain_id),
    INDEX idx_is_match (is_match),
    INDEX idx_domain (domain_id),
    INDEX idx_news (news_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='热点新闻领域分析结果表';

-- 提取的关键词表
CREATE TABLE IF NOT EXISTS extracted_keywords (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    keyword VARCHAR(200) NOT NULL UNIQUE COMMENT '提取的关键词',
    source_news_id VARCHAR(100) COMMENT '来源热点新闻的news_id',
    domain_id INT COMMENT '关联的领域配置id',
    llm_provider VARCHAR(50) COMMENT '使用的LLM提供商',
    confidence DECIMAL(5,2) COMMENT '关键词置信度/权重',
    search_count INT DEFAULT 0 COMMENT '被用于搜索爬取的次数',
    last_used DATETIME COMMENT '最后使用时间',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '提取时间',
    INDEX idx_keyword (keyword),
    INDEX idx_domain (domain_id),
    INDEX idx_search (search_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='从热点新闻中提取的关键词表';

-- 关键词爬取记录表
CREATE TABLE IF NOT EXISTS keyword_crawl_log (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    keyword_id INT COMMENT '关联extracted_keywords表的主键id',
    platform_code VARCHAR(50) COMMENT '爬取的目标平台代码',
    status VARCHAR(20) COMMENT '爬取状态：pending-待处理，processing-处理中，completed-已完成，failed-失败',
    notes_count INT DEFAULT 0 COMMENT '成功爬取的笔记/视频数量',
    comments_count INT DEFAULT 0 COMMENT '成功爬取的评论数量',
    error_message TEXT COMMENT '错误信息，失败时记录具体错误原因',
    started_at DATETIME COMMENT '爬取任务开始时间',
    completed_at DATETIME COMMENT '爬取任务完成时间',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
    INDEX idx_keyword (keyword_id),
    INDEX idx_status (status),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='关键词爬取任务执行记录表';

-- 任务执行日志表
CREATE TABLE IF NOT EXISTS task_execution_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    task_name VARCHAR(100) NOT NULL COMMENT '任务名称',
    status VARCHAR(20) NOT NULL COMMENT '执行状态：running-运行中，success-成功，failed-失败',
    hot_count INT DEFAULT 0 COMMENT '本次任务获取的热点新闻数量',
    matched_count INT DEFAULT 0 COMMENT '本次任务匹配领域的热点数量',
    keyword_count INT DEFAULT 0 COMMENT '本次任务提取的关键词数量',
    crawl_triggered INT DEFAULT 0 COMMENT '本次任务触发爬取的次数',
    error_message TEXT COMMENT '错误信息，失败时记录具体错误原因',
    started_at DATETIME COMMENT '任务开始时间',
    completed_at DATETIME COMMENT '任务完成时间',
    INDEX idx_task_time (task_name, started_at),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务执行日志表，记录每次任务执行的详细信息';

-- =====================================================
-- 初始化默认数据
-- =====================================================

-- 热点平台默认配置
INSERT INTO hot_platform_config (platform_code, platform_name, priority) VALUES
('baidu', '百度热搜', 10),
('weibo', '微博热搜', 20),
('zhihu', '知乎热榜', 15),
('bilibili', 'B站热门', 12),
('douyin', '抖音热点', 18),
('juejin', '掘金技术', 8),
('github', 'GitHub Trending', 5),
('hackernews', 'Hacker News', 3),
('sina_finance', '新浪财经', 7),
('xueqiu', '雪球', 6)
ON DUPLICATE KEY UPDATE platform_name=VALUES(platform_name);

-- 任务调度默认配置
INSERT INTO task_schedule_config (task_name, cron_expression, is_enabled, max_hot_count) VALUES
('hot_fetch', '0 */2 * * *', 1, 100),
('hot_analyze', '5 */2 * * *', 1, 100),
('hot_extract', '10 */2 * * *', 1, 100),
('hot_crawl', '15 */2 * * *', 1, 50)
ON DUPLICATE KEY UPDATE cron_expression=VALUES(cron_expression);

-- 爬虫平台默认配置
INSERT INTO crawler_platform_config (platform_code, platform_name, is_enabled, max_notes_count) VALUES
('xhs', '小红书', 1, 30),
('dy', '抖音', 1, 30),
('ks', '快手', 0, 20),
('bili', 'B站', 1, 20),
('wb', '微博', 1, 20),
('tieba', '贴吧', 0, 20),
('zhihu', '知乎', 0, 20)
ON DUPLICATE KEY UPDATE platform_name=VALUES(platform_name);
