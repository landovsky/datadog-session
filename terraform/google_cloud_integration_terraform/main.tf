provider "google" {
  project = var.gcp_project_id
  region  = "us-central1"
  zone    = "us-central1-c"
}

resource "google_service_account" "datadog_integration" {
  account_id   = "datadog-${var.gcp_project_id}"
  display_name = "Datadog Integration"
  project      = var.gcp_project_id
}

# Roles for the Datadog integration SA (used by Datadog for GCP metrics and Vertex AI)
# aiplatform.user is granted separately with a condition so only specific models are allowed.
locals {
  datadog_roles = [
    "roles/compute.viewer",
    "roles/monitoring.viewer",
    "roles/cloudasset.viewer",
    "roles/browser",
    "roles/serviceusage.serviceUsageConsumer",
    "roles/aiplatform.user",
  ]
}

# Assign the required roles to the service account
resource "google_project_iam_member" "datadog_roles" {
  for_each = toset(local.datadog_roles)
  project  = var.gcp_project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.datadog_integration.email}"
}

variable "gcp_labels" {
  description = "Common labels for all resources deployed by this module"
  type        = map(string)
  default     = {
    application = "datadog-integration"
  }
}
