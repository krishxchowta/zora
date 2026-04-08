alter table public.runs
  add column if not exists feature_summary jsonb;
