variable "gcp_project_id" {
  description = "The project ID for our website infra."
  type        = string
}

variable "datadog_app_key" {
  type        = string
  sensitive   = true
}

variable "datadog_api_key" {
  type        = string
  sensitive   = true
}

