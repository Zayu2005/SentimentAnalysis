-- ============================================================
-- 006_topic_classification.sql
-- 话题多维分类标签系统
-- 支持一级分类、事件类型、受众范围、风险等级等多维度分类
-- ============================================================

-- ==================== 1. 话题分类表 ====================
-- 存储话题的多维分类标签

CREATE TABLE IF NOT EXISTS topic_classification (
    -- ==================== 主键与关联 ====================
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '分类记录自增主键',
    topic_id BIGINT NOT NULL COMMENT '关联话题事件ID',

    -- ==================== 多级分类 ====================
    primary_category VARCHAR(50) COMMENT '一级分类(大领域)',
    secondary_category VARCHAR(100) COMMENT '二级分类(具体领域)',
    tertiary_category VARCHAR(100) COMMENT '三级分类(细分场景)',

    -- ==================== 行业标签 (多值) ====================
    industry_tags JSON COMMENT '行业标签列表,如["汽车","新能源"]',

    -- ==================== 事件属性 ====================
    event_type VARCHAR(50) COMMENT '事件类型',
    audience_scope ENUM('local', 'regional', 'national', 'global')
        DEFAULT 'national' COMMENT '受众范围',
    time_sensitivity ENUM('breaking', 'trending', 'normal', 'evergreen')
        DEFAULT 'normal' COMMENT '时间敏感度',

    -- ==================== 情感属性 ====================
    sentiment_intensity ENUM('mild', 'moderate', 'intense', 'extreme')
        DEFAULT 'moderate' COMMENT '情感强度',
    controversy_level ENUM('low', 'medium', 'high', 'extreme')
        DEFAULT 'low' COMMENT '争议程度',

    -- ==================== 风险评估 ====================
    risk_level ENUM('safe', 'attention', 'warning', 'dangerous')
        DEFAULT 'safe' COMMENT '舆情风险等级',
    risk_keywords JSON COMMENT '风险关键词',

    -- ==================== 分类置信度 ====================
    classification_confidence DECIMAL(5,4) COMMENT 'LLM分类置信度(0-1)',
    classified_by VARCHAR(50) DEFAULT 'llm' COMMENT '分类来源: llm/rule/manual',
    classification_reason TEXT COMMENT '分类理由',

    -- ==================== 系统时间 ====================
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '分类时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    -- ==================== 约束与索引 ====================
    UNIQUE KEY uk_topic_id (topic_id),
    INDEX idx_primary_category (primary_category),
    INDEX idx_event_type (event_type),
    INDEX idx_risk_level (risk_level),
    INDEX idx_time_sensitivity (time_sensitivity),
    INDEX idx_classified_by (classified_by),

    CONSTRAINT fk_classification_topic FOREIGN KEY (topic_id)
        REFERENCES topic_event(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT '话题分类表 - 存储话题的多维分类标签';


-- ==================== 2. 扩展 topic_event 表 ====================
-- 添加分类相关冗余字段，便于快速查询

ALTER TABLE topic_event
    ADD COLUMN primary_category VARCHAR(50) COMMENT '一级分类(冗余字段)'
        AFTER dominant_emotions,
    ADD COLUMN event_type VARCHAR(50) COMMENT '事件类型(冗余字段)'
        AFTER primary_category,
    ADD COLUMN risk_level ENUM('safe', 'attention', 'warning', 'dangerous')
        DEFAULT 'safe' COMMENT '舆情风险等级(冗余字段)'
        AFTER event_type,
    ADD COLUMN classification_version INT DEFAULT 0 COMMENT '分类版本号'
        AFTER risk_level;


-- ==================== 3. 视图: 话题分类全景 ====================
-- 关联话题表和分类表，提供完整的分类信息

CREATE OR REPLACE VIEW v_topic_classification AS
SELECT
    te.id AS topic_id,
    te.event_name,
    te.status,
    te.heat_level,
    te.content_count,
    te.avg_sentiment_score,
    te.dominant_sentiment,
    -- 分类信息
    tc.primary_category,
    tc.secondary_category,
    tc.tertiary_category,
    tc.industry_tags,
    tc.event_type,
    tc.audience_scope,
    tc.time_sensitivity,
    tc.sentiment_intensity,
    tc.controversy_level,
    tc.risk_level,
    tc.classification_confidence,
    tc.classified_by,
    -- 时间
    te.first_content_at,
    te.last_content_at,
    te.created_at AS topic_created_at,
    tc.created_at AS classified_at
FROM topic_event te
LEFT JOIN topic_classification tc ON te.id = tc.topic_id
WHERE te.status != 'merged';


-- ==================== 4. 视图: 高风险话题监控 ====================
-- 用于快速筛选需要关注的高风险话题

CREATE OR REPLACE VIEW v_high_risk_topics AS
SELECT
    te.id AS topic_id,
    te.event_name,
    te.event_description,
    te.content_count,
    te.heat_level,
    te.avg_sentiment_score,
    te.dominant_sentiment,
    te.risk_level,
    tc.primary_category,
    tc.event_type,
    tc.controversy_level,
    tc.sentiment_intensity,
    tc.classification_confidence,
    te.first_content_at,
    te.last_content_at
FROM topic_event te
LEFT JOIN topic_classification tc ON te.id = tc.topic_id
WHERE te.status != 'merged'
  AND te.risk_level IN ('warning', 'dangerous')
   OR tc.risk_level IN ('warning', 'dangerous')
ORDER BY
    FIELD(te.risk_level, 'dangerous', 'warning', 'attention', 'safe') ASC,
    te.content_count DESC;


-- ==================== 5. 统计存储过程 ====================

DELIMITER //

DROP PROCEDURE IF EXISTS sp_update_topic_classification_stats//

CREATE PROCEDURE sp_update_topic_classification_stats()
BEGIN
    -- 将 topic_classification 表的信息同步回 topic_event 表
    UPDATE topic_event te
    JOIN topic_classification tc ON te.id = tc.topic_id
    SET
        te.primary_category = tc.primary_category,
        te.event_type = tc.event_type,
        te.risk_level = tc.risk_level,
        te.classification_version = te.classification_version + 1
    WHERE tc.classified_by = 'llm'
      AND te.classification_version = 0;
END//

DELIMITER ;
