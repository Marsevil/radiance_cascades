use std::sync::Arc;

use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::Surface,
    VulkanLibrary,
};
use winit::window::Window;

fn get_vulkan_instance(window: &Window) -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("Vulkan library not found");
    let extensions = Surface::required_extensions(window);
    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: extensions,
            ..Default::default()
        },
    )
    .unwrap()
}

fn get_best_physical_device(
    instance: &Arc<Instance>,
    surface: &Surface,
    enabled_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("Can't enumerate devices")
        .filter(|device| device.supported_extensions().contains(enabled_extensions))
        .filter_map(|device| {
            device
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(queue_idx, queue)| {
                    queue.queue_flags.contains(QueueFlags::GRAPHICS)
                        && device
                            .surface_support(queue_idx as u32, surface)
                            .unwrap_or(false)
                })
                .map(|q| (device, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("No physical device found")
}

fn get_logical_device(
    physical_device: Arc<PhysicalDevice>,
    enabled_extensions: DeviceExtensions,
    queue_family_idx: u32,
) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: queue_family_idx,
                ..Default::default()
            }],
            enabled_extensions,
            ..Default::default()
        },
    )
    .expect("Failed to create device");
    let queue = queues.next().expect("No available queue found");

    (device, queue)
}

/// This struct contains all parts of vulkan API that are not changing.
pub struct VulkanCore {
    pub instance: Arc<Instance>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
}
impl VulkanCore {
    pub fn new(window: Arc<Window>) -> Self {
        let instance = get_vulkan_instance(&window);
        let surface = Surface::from_window(instance.clone(), window)
            .expect("Unable to create vulkan surface");
        let enabled_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        };
        let (physical_device, queue_family_idx) =
            get_best_physical_device(&instance, &surface, &enabled_extensions);
        let (device, queue) = get_logical_device(
            physical_device.clone(),
            enabled_extensions,
            queue_family_idx,
        );

        Self {
            instance,
            physical_device,
            device,
            queue,
            surface,
        }
    }
}
