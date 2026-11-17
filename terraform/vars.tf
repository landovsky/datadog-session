variable "instruqt_participant_id" {
  description = "Unique id for instruqt participant in instruqt track."
  type        = string
}

//Datadog Vars

variable "datadog_app_key" {
  type        = string
  sensitive   = true
}

variable "datadog_api_key" {
  type        = string
  sensitive   = true
}

//GCP Vars

variable "gcp_project_id" {
  description = "The project ID for instruqt sandbox GCP project."
  type        = string
}

