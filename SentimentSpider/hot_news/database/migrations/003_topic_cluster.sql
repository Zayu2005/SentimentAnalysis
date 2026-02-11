-- ============================================================
-- 003_topic_cluster.sql
-- 话题聚类与事件检测模块
-- 基于 HISEvent (AAAI 2024) + RagSEDE (2026) 设计思路
-- ============================================================

-- ==================== 1. 话题事件表 ====================
-- 每个话题即为一个事件，由增量聚类自动发现
-- 质心嵌入用于增量匹配，话题描述由 LLM 生成

CREATE TABLE IF NOT EXISTS topic_event (
    -- ==================== 主键与标识 ====================
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '话题事件自增主键',

    -- ==================== 话题描述（LLM 生成） ====================
    event_name VARCHAR(200) NOT NULL COMMENT '话题名称(LLM生成, 如"某品牌售后服务争议")',
    event_description TEXT COMMENT '话题详细描述(LLM生成)',
    keywords JSON COMMENT '话题关键词及权重, 如[{"word":"售后","weight":0.9}]',

    -- ==================== 向量表示 ====================
    centroid_embedding BLOB COMMENT '话题质心嵌入向量(float32序列化, 用于增量匹配)',
    embedding_model VARCHAR(100) DEFAULT 'hfl/chinese-roberta-wwm-ext' COMMENT '嵌入模型名称',

    -- ==================== 话题状态 ====================
    status ENUM('emerging', 'active', 'declining', 'ended', 'merged')
        DEFAULT 'emerging' COMMENT '话题状态: emerging新兴/active活跃/declining衰退/ended结束/merged已合并',
    merged_into_id BIGINT COMMENT '合并目标话题ID(status=merged时有值)',
    heat_level ENUM('low', 'medium', 'high', 'critical')
        DEFAULT 'low' COMMENT '热度等级',

    -- ==================== 统计数据（缓存，定期更新） ====================
    content_count INT DEFAULT 0 COMMENT '关联内容数量',
    comment_count INT DEFAULT 0 COMMENT '关联评论数量(通过内容间接统计)',
    platform_distribution JSON COMMENT '平台分布, 如{"xhs":120,"wb":85,"dy":60}',

    -- ==================== 情感聚合 ====================
    avg_sentiment_score DECIMAL(5,4) COMMENT '平均情感得分(-1到1)',
    sentiment_distribution JSON COMMENT '情感分布, 如{"positive":30,"neutral":50,"negative":20}',
    dominant_sentiment ENUM('positive', 'negative', 'neutral', 'mixed')
        COMMENT '主导情感倾向',
    dominant_emotions VARCHAR(200) COMMENT '主要情绪标签(如"愤怒,失望")',

    -- ==================== 时间线 ====================
    first_content_at DATETIME COMMENT '最早内容发布时间',
    last_content_at DATETIME COMMENT '最新内容发布时间',
    peak_at DATETIME COMMENT '话题高峰时间(内容最密集)',

    -- ==================== 聚类参数 ====================
    similarity_threshold DECIMAL(5,4) DEFAULT 0.7500 COMMENT '该话题使用的相似度阈值',
    cluster_version INT DEFAULT 1 COMMENT '聚类版本(全量重聚类时递增)',

    -- ==================== 系统时间 ====================
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '话题创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '话题更新时间',

    -- ==================== 索引 ====================
    INDEX idx_status (status) COMMENT '按话题状态筛选(emerging/active/declining/ended/merged)',
    INDEX idx_heat_level (heat_level) COMMENT '按热度等级筛选(low/medium/high/critical)',
    INDEX idx_dominant_sentiment (dominant_sentiment) COMMENT '按主导情感筛选',
    INDEX idx_first_content_at (first_content_at) COMMENT '按话题起始时间排序和范围查询',
    INDEX idx_last_content_at (last_content_at) COMMENT '按最新内容时间排序, 用于活跃度判断',
    INDEX idx_created_at (created_at) COMMENT '按话题创建时间排序',
    INDEX idx_content_count (content_count) COMMENT '按内容数量排序, 用于话题规模筛选',
    INDEX idx_merged_into (merged_into_id) COMMENT '查找合并到指定话题的所有源话题'

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT '话题事件表 - 增量聚类发现的话题, 每个话题即为一个事件';


-- ==================== 2. 话题演化快照表 ====================
-- 按天记录话题的状态变化，用于追踪话题生命周期和趋势分析

CREATE TABLE IF NOT EXISTS topic_evolution (
    -- ==================== 主键与标识 ====================
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '快照自增主键',
    event_id BIGINT NOT NULL COMMENT '关联话题事件ID',
    snapshot_date DATE NOT NULL COMMENT '快照日期',

    -- ==================== 当日增量 ====================
    content_count_delta INT DEFAULT 0 COMMENT '当日新增内容数',
    comment_count_delta INT DEFAULT 0 COMMENT '当日新增评论数',
    content_count_total INT DEFAULT 0 COMMENT '截至当日累计内容数',

    -- ==================== 当日情感 ====================
    avg_sentiment_score DECIMAL(5,4) COMMENT '当日平均情感得分',
    sentiment_distribution JSON COMMENT '当日情感分布',

    -- ==================== 热度指标 ====================
    hot_score DECIMAL(10,4) DEFAULT 0.0000 COMMENT '当日热度分数(综合互动量、增速等)',
    interaction_count INT DEFAULT 0 COMMENT '当日总互动量(点赞+评论+分享)',

    -- ==================== 当日关键词 ====================
    keywords JSON COMMENT '当日关键词(可能随时间漂移)',
    platform_distribution JSON COMMENT '当日各平台分布',

    -- ==================== 系统时间 ====================
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '快照创建时间',

    -- ==================== 索引 ====================
    UNIQUE KEY uk_event_date (event_id, snapshot_date) COMMENT '同一话题同一天仅保留一条快照(UPSERT)',
    INDEX idx_snapshot_date (snapshot_date) COMMENT '按日期查询全局快照',
    INDEX idx_hot_score (hot_score) COMMENT '按热度分数排序, 用于热门话题排行',

    -- ==================== 外键 ====================
    CONSTRAINT fk_evolution_event FOREIGN KEY (event_id)
        REFERENCES topic_event(id) ON DELETE CASCADE

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT '话题演化快照表 - 按天追踪话题状态变化与趋势';


-- ==================== 3. 话题合并记录表 ====================
-- 记录话题合并历史，便于溯源和审计

CREATE TABLE IF NOT EXISTS topic_merge_log (
    -- ==================== 主键与标识 ====================
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '合并记录自增主键',

    -- ==================== 合并信息 ====================
    source_event_id BIGINT NOT NULL COMMENT '被合并的话题ID',
    target_event_id BIGINT NOT NULL COMMENT '合并目标话题ID',
    similarity_score DECIMAL(5,4) COMMENT '合并时的相似度分数',
    merge_reason VARCHAR(200) COMMENT '合并原因(如"语义高度重合","质心距离<0.15")',

    -- ==================== 合并前快照 ====================
    source_content_count INT COMMENT '被合并话题当时的内容数',
    source_keywords JSON COMMENT '被合并话题当时的关键词',

    -- ==================== 系统时间 ====================
    merged_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '合并时间',

    -- ==================== 索引 ====================
    INDEX idx_source_event (source_event_id) COMMENT '查找指定话题被合并的记录',
    INDEX idx_target_event (target_event_id) COMMENT '查找合并到指定话题的所有来源',
    INDEX idx_merged_at (merged_at) COMMENT '按合并时间排序, 用于审计追溯'

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT '话题合并记录表 - 记录话题合并历史便于溯源';


-- ==================== 4. 扩展 unified_content 表 ====================
-- 为统一内容表添加话题聚类字段

ALTER TABLE unified_content
    ADD COLUMN topic_id BIGINT COMMENT '所属话题事件ID, 关联topic_event.id'
        AFTER sentiment_analyzed_at,
    ADD COLUMN topic_similarity DECIMAL(5,4) COMMENT '与话题质心的余弦相似度(0到1, 越大越相关)'
        AFTER topic_id,
    ADD COLUMN topic_assigned_at DATETIME COMMENT '话题分配时间, 由聚类引擎写入'
        AFTER topic_similarity,
    ADD INDEX idx_topic_id (topic_id) COMMENT '按话题ID查询该话题下所有内容',
    ADD INDEX idx_topic_assigned (topic_assigned_at) COMMENT '按分配时间查询, 用于增量统计';


-- ==================== 5. 视图: 话题事件全景 ====================
-- 便于快速查询话题事件的完整信息
-- 排除已合并话题, 按最新内容时间倒序
-- 计算 duration_days 表示话题持续天数

CREATE OR REPLACE VIEW v_topic_overview AS
SELECT
    te.id AS event_id,                                                  -- 话题ID
    te.event_name,                                                      -- 话题名称
    te.status,                                                          -- 话题状态
    te.heat_level,                                                      -- 热度等级
    te.content_count,                                                   -- 关联内容数
    te.comment_count,                                                   -- 关联评论数
    te.avg_sentiment_score,                                             -- 平均情感得分
    te.dominant_sentiment,                                              -- 主导情感
    te.dominant_emotions,                                               -- 主要情绪标签
    te.platform_distribution,                                           -- 平台分布JSON
    te.first_content_at,                                                -- 最早内容时间
    te.last_content_at,                                                 -- 最新内容时间
    te.peak_at,                                                         -- 高峰时间
    DATEDIFF(te.last_content_at, te.first_content_at) AS duration_days, -- 话题持续天数
    te.created_at                                                       -- 话题创建时间
FROM topic_event te
WHERE te.status != 'merged'
ORDER BY te.last_content_at DESC;


-- ==================== 6. 视图: 话题情感分析 ====================
-- 每个话题下各平台的情感分布
-- 按 (话题ID, 话题名称, 平台) 分组统计正/中/负面内容数和平均分

CREATE OR REPLACE VIEW v_topic_sentiment AS
SELECT
    te.id AS event_id,                                                              -- 话题ID
    te.event_name,                                                                  -- 话题名称
    uc.platform,                                                                    -- 来源平台
    COUNT(*) AS content_count,                                                      -- 该平台内容数
    AVG(uc.sentiment_score) AS avg_score,                                           -- 该平台平均情感分
    SUM(CASE WHEN uc.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_count,  -- 正面内容数
    SUM(CASE WHEN uc.sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral_count,    -- 中性内容数
    SUM(CASE WHEN uc.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_count   -- 负面内容数
FROM topic_event te
JOIN unified_content uc ON uc.topic_id = te.id
WHERE te.status != 'merged'
GROUP BY te.id, te.event_name, uc.platform;


-- ==================== 7. 视图: 评论通过内容关联话题 ====================
-- 评论不单独聚类，通过 content_id 继承所属内容的话题
-- JOIN 路径: topic_event → unified_content → unified_comment

CREATE OR REPLACE VIEW v_topic_comments AS
SELECT
    te.id AS event_id,                  -- 话题ID
    te.event_name,                      -- 话题名称
    ucmt.id AS comment_id,              -- 评论ID
    ucmt.platform,                      -- 评论来源平台
    ucmt.content,                       -- 评论内容
    ucmt.sentiment,                     -- 评论情感标签
    ucmt.sentiment_score,               -- 评论情感分数
    ucmt.original_created_at            -- 评论原始发布时间
FROM topic_event te
JOIN unified_content uc ON uc.topic_id = te.id
JOIN unified_comment ucmt ON ucmt.platform = uc.platform AND ucmt.content_id = uc.content_id
WHERE te.status != 'merged';
