use thiserror::Error;
use vulkano::VulkanError;

#[derive(Debug, Error)]
pub enum DrawingError {
    #[error("Swapchain needs to be recreate")]
    ObsoleteSwapchain,
    #[error("Uncatched vulkan error")]
    VulkanError(VulkanError),
}

impl From<VulkanError> for DrawingError {
    fn from(value: VulkanError) -> Self {
        match value {
            VulkanError::OutOfDate => Self::ObsoleteSwapchain,
            _ => Self::VulkanError(value),
        }
    }
}
