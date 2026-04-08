-- Doctor clinical board workflow schema
-- Apply this in Supabase SQL editor before using the board routes in production.

create table if not exists public.patient_contacts (
  id bigserial primary key,
  run_id text not null,
  patient_name text not null,
  phone_e164 text,
  whatsapp_e164 text,
  preferred_channel text not null default 'whatsapp',
  created_at timestamptz not null default now()
);

create table if not exists public.report_requests (
  id bigserial primary key,
  run_id text not null,
  patient_contact_id bigint not null references public.patient_contacts(id) on delete cascade,
  request_message text,
  status text not null default 'requested',
  requested_at timestamptz not null default now(),
  doctor_notified_at timestamptz,
  approved_at timestamptz,
  rejected_at timestamptz,
  sent_at timestamptz,
  doctor_name text
);

create table if not exists public.prescriptions (
  id bigserial primary key,
  run_id text not null,
  report_request_id bigint references public.report_requests(id) on delete cascade,
  doctor_name text not null,
  prescription_text text not null,
  notes text,
  is_final boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.message_deliveries (
  id bigserial primary key,
  run_id text not null,
  report_request_id bigint references public.report_requests(id) on delete cascade,
  channel text not null,
  recipient_role text not null,
  recipient_e164 text not null,
  message_type text not null,
  delivery_status text not null,
  provider_message_id text,
  error_text text,
  sent_at timestamptz not null default now()
);

alter table public.patient_contacts
  add column if not exists run_id text,
  add column if not exists patient_name text,
  add column if not exists phone_e164 text,
  add column if not exists whatsapp_e164 text,
  add column if not exists preferred_channel text default 'whatsapp',
  add column if not exists created_at timestamptz default now();

alter table public.report_requests
  add column if not exists run_id text,
  add column if not exists patient_contact_id bigint,
  add column if not exists request_message text,
  add column if not exists status text default 'requested',
  add column if not exists requested_at timestamptz default now(),
  add column if not exists doctor_notified_at timestamptz,
  add column if not exists approved_at timestamptz,
  add column if not exists rejected_at timestamptz,
  add column if not exists sent_at timestamptz,
  add column if not exists doctor_name text;

alter table public.prescriptions
  add column if not exists run_id text,
  add column if not exists report_request_id bigint,
  add column if not exists doctor_name text,
  add column if not exists prescription_text text,
  add column if not exists notes text,
  add column if not exists is_final boolean default false,
  add column if not exists created_at timestamptz default now(),
  add column if not exists updated_at timestamptz default now();

alter table public.message_deliveries
  add column if not exists run_id text,
  add column if not exists report_request_id bigint,
  add column if not exists channel text,
  add column if not exists recipient_role text,
  add column if not exists recipient_e164 text,
  add column if not exists message_type text,
  add column if not exists delivery_status text,
  add column if not exists provider_message_id text,
  add column if not exists error_text text,
  add column if not exists sent_at timestamptz default now();

alter table public.insights
  add column if not exists doctor_report_text text,
  add column if not exists patient_report_text text,
  add column if not exists final_prescription_text text,
  add column if not exists report_status text;

create index if not exists idx_patient_contacts_run_id
  on public.patient_contacts(run_id);

create index if not exists idx_report_requests_run_id
  on public.report_requests(run_id);

create index if not exists idx_report_requests_status
  on public.report_requests(status);

create index if not exists idx_prescriptions_run_id
  on public.prescriptions(run_id);

create index if not exists idx_prescriptions_report_request_id
  on public.prescriptions(report_request_id);

create index if not exists idx_message_deliveries_run_id
  on public.message_deliveries(run_id);

create index if not exists idx_message_deliveries_report_request_id
  on public.message_deliveries(report_request_id);
