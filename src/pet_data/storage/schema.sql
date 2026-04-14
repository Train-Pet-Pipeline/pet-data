CREATE TABLE IF NOT EXISTS frames (
    frame_id        TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL,
    source          TEXT NOT NULL,
    frame_path      TEXT NOT NULL,
    data_root       TEXT NOT NULL,
    timestamp_ms    INTEGER,
    species         TEXT,
    breed           TEXT,
    lighting        TEXT CHECK(lighting IN ('bright','dim','infrared_night','unknown')),
    bowl_type       TEXT,
    quality_flag    TEXT NOT NULL DEFAULT 'normal'
                    CHECK(quality_flag IN ('normal','low','failed')),
    blur_score      REAL,
    phash           BLOB,
    aug_quality     TEXT CHECK(aug_quality IN ('ok','failed') OR aug_quality IS NULL),
    aug_seed        INTEGER,
    parent_frame_id TEXT,
    is_anomaly_candidate INTEGER NOT NULL DEFAULT 0,
    anomaly_score   REAL,
    annotation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(annotation_status IN ('pending','annotating','auto_checked',
                                    'approved','needs_review','reviewed','rejected','exported')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_frames_status    ON frames(annotation_status);
CREATE INDEX IF NOT EXISTS idx_frames_source    ON frames(source);
CREATE INDEX IF NOT EXISTS idx_frames_quality   ON frames(quality_flag);
CREATE INDEX IF NOT EXISTS idx_frames_anomaly   ON frames(is_anomaly_candidate, anomaly_score DESC);
