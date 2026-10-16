// The Vulkan compatible tier: the same API and model on Vulkan 1.3 GPUs that lack VK_EXT_descriptor_heap,
// VK_KHR_device_address_commands, VK_KHR_shader_untyped_pointers or VK_EXT_mesh_shader, such as a GTX 10, an RX 5000 or
// 6000, Arc or Iris Xe. NoGraphicsAPI.cpp includes this file inside namespace gpu, and the call sites that differ by tier
// call these functions. See docs/vulkan-support.md.

namespace {
    namespace compatible {

        // The bindings of the device's one descriptor set, and the stages that read them and the root.
        constexpr uint32 sampled_texture_binding = 0;
        constexpr uint32 storage_texture_binding = 1;
        constexpr uint32 sampler_binding = 2;
        constexpr VkShaderStageFlags root_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

        // Where a GPU address is to the classic buffer commands.
        struct BufferLocation
        {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceSize offset = 0;
        };

        // Adds the name of each thing the compatible tier needs, beyond what both tiers need, that the device lacks.
        void note_missing_requirements(
            const QueriedFeatures& features,
            const VkPhysicalDeviceVulkan12Properties& limits,
            const DeviceDesc& desc,
            MissingRequirements& missing
        ) noexcept
        {
            const RequiredFeature descriptor_indexing[]{
                { .supported = features.vulkan12.runtimeDescriptorArray, .name = "runtimeDescriptorArray" },
                { .supported = features.vulkan12.descriptorBindingPartiallyBound, .name = "descriptorBindingPartiallyBound" },
                { .supported = features.vulkan12.descriptorBindingSampledImageUpdateAfterBind, .name = "descriptorBindingSampledImageUpdateAfterBind" },
                { .supported = features.vulkan12.descriptorBindingStorageImageUpdateAfterBind, .name = "descriptorBindingStorageImageUpdateAfterBind" },
                { .supported = features.vulkan12.shaderSampledImageArrayNonUniformIndexing, .name = "shaderSampledImageArrayNonUniformIndexing" },
            };
            for (const RequiredFeature& feature : descriptor_indexing)
            {
                if (feature.supported != VK_TRUE)
                    note_missing(missing, feature.name);
            }

            const uint32 textures = desc.compatible_texture_descriptor_capacity;
            const uint32 samplers = desc.compatible_sampler_descriptor_capacity;
            const bool has_room_for_sampled_images =
                limits.maxPerStageDescriptorUpdateAfterBindSampledImages >= textures && limits.maxDescriptorSetUpdateAfterBindSampledImages >= textures;
            const bool has_room_for_storage_images =
                limits.maxPerStageDescriptorUpdateAfterBindStorageImages >= textures && limits.maxDescriptorSetUpdateAfterBindStorageImages >= textures;
            const bool has_room_for_samplers =
                limits.maxPerStageDescriptorUpdateAfterBindSamplers >= samplers && limits.maxDescriptorSetUpdateAfterBindSamplers >= samplers;
            const uint64 descriptors_a_stage_reads = 2ull * textures + samplers;
            const bool has_room_for_every_descriptor = limits.maxPerStageUpdateAfterBindResources >= descriptors_a_stage_reads;
            if (!has_room_for_sampled_images)
                note_missing(missing, "room for the sampled image descriptors");
            if (!has_room_for_storage_images)
                note_missing(missing, "room for the storage image descriptors");
            if (!has_room_for_samplers)
                note_missing(missing, "room for the sampler descriptors");
            if (!has_room_for_every_descriptor)
                note_missing(missing, "room for every descriptor in one stage");
        }

        void enable_features(QueriedFeatures& features) noexcept
        {
            features.vulkan12.runtimeDescriptorArray = VK_TRUE;
            features.vulkan12.descriptorBindingPartiallyBound = VK_TRUE;
            features.vulkan12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
            features.vulkan12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
            features.vulkan12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        }

        // The stand-in for the descriptor heaps, sized by DeviceDesc's compatible capacities. Every binding is update-after-bind,
        // so slots can be written while frames that use the set are in flight, and partially bound, so unwritten slots are fine.
        Error create_descriptors(Device& device, const DeviceDesc& desc) noexcept
        {
            const uint32 textures = desc.compatible_texture_descriptor_capacity;
            const uint32 samplers = desc.compatible_sampler_descriptor_capacity;
            const VkDescriptorSetLayoutBinding bindings[]{
                {
                    .binding = sampled_texture_binding,
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    .descriptorCount = textures,
                    .stageFlags = root_stages,
                },
                {
                    .binding = storage_texture_binding,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = textures,
                    .stageFlags = root_stages,
                },
                {
                    .binding = sampler_binding,
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                    .descriptorCount = samplers,
                    .stageFlags = root_stages,
                },
            };
            constexpr VkDescriptorBindingFlags each_binding = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            const VkDescriptorBindingFlags binding_flags[]{ each_binding, each_binding, each_binding };
            const VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                .bindingCount = 3,
                .pBindingFlags = binding_flags,
            };
            const VkDescriptorSetLayoutCreateInfo layout_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = &binding_flags_info,
                .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                .bindingCount = 3,
                .pBindings = bindings,
            };
            detail::CompatibleDescriptors& descriptors = device.compatible_descriptors;
            Error error = error_from_vk(vkCreateDescriptorSetLayout(device.device, &layout_info, nullptr, &descriptors.layout), "vkCreateDescriptorSetLayout");
            if (error != Error::none)
                return error;

            const VkDescriptorPoolSize pool_sizes[]{
                {
                    .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    .descriptorCount = textures,
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = textures,
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_SAMPLER,
                    .descriptorCount = samplers,
                },
            };
            const VkDescriptorPoolCreateInfo pool_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                .maxSets = 1,
                .poolSizeCount = 3,
                .pPoolSizes = pool_sizes,
            };
            error = error_from_vk(vkCreateDescriptorPool(device.device, &pool_info, nullptr, &descriptors.pool), "vkCreateDescriptorPool");
            if (error != Error::none)
                return error;

            const VkDescriptorSetAllocateInfo set_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptors.pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptors.layout,
            };
            error = error_from_vk(vkAllocateDescriptorSets(device.device, &set_info, &descriptors.set), "vkAllocateDescriptorSets");
            if (error != Error::none)
                return error;

            const VkPushConstantRange root_address{
                .stageFlags = root_stages,
                .offset = 0,
                .size = sizeof(void*),
            };
            const VkPipelineLayoutCreateInfo pipeline_layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &descriptors.layout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &root_address,
            };
            return error_from_vk(vkCreatePipelineLayout(device.device, &pipeline_layout_info, nullptr, &descriptors.pipeline_layout), "vkCreatePipelineLayout");
        }

        // Destroying the pool frees the set.
        void destroy_descriptors(Device& device) noexcept
        {
            const detail::CompatibleDescriptors& descriptors = device.compatible_descriptors;
            vkDestroyPipelineLayout(device.device, descriptors.pipeline_layout, nullptr);
            vkDestroyDescriptorPool(device.device, descriptors.pool, nullptr);
            vkDestroyDescriptorSetLayout(device.device, descriptors.layout, nullptr);
        }

        void add_heap_buffer(Device& device, const detail::BackingBuffer& backing, VkDeviceSize byte_count) noexcept
        {
            assert(device.heap_buffer_count < max_heap_buffers && "more GPU heaps are alive than max_heap_buffers");
            device.heap_buffers[device.heap_buffer_count] = {
                .first_address = backing.address,
                .byte_count = byte_count,
                .buffer = backing.buffer,
            };
            ++device.heap_buffer_count;
            if (device.heap_buffer_count > device.heap_buffer_high_water)
                device.heap_buffer_high_water = device.heap_buffer_count;
        }

        // The last entry takes the removed one's place.
        void remove_heap_buffer(Device& device, VkBuffer buffer) noexcept
        {
            for (uint32 index = 0; index < device.heap_buffer_count; ++index)
            {
                if (device.heap_buffers[index].buffer != buffer)
                    continue;
                --device.heap_buffer_count;
                device.heap_buffers[index] = device.heap_buffers[device.heap_buffer_count];
                return;
            }
        }

        // Every address a command is given lies in a live GPU heap.
        BufferLocation find_heap_buffer(const Device& device, const void* address) noexcept
        {
            const VkDeviceAddress wanted = static_cast<VkDeviceAddress>(reinterpret_cast<uintptr>(address));
            for (uint32 index = 0; index < device.heap_buffer_count; ++index)
            {
                const detail::HeapBuffer& heap_buffer = device.heap_buffers[index];
                // Unsigned: an address below the block wraps past its size, so one comparison covers both ends.
                const VkDeviceSize offset = wanted - heap_buffer.first_address;
                if (offset < heap_buffer.byte_count)
                    return { .buffer = heap_buffer.buffer, .offset = offset };
            }
            assert(false && "the address is in no live GPU heap");
            return {};
        }

        TextureDescriptorHeap* create_texture_descriptor_heap(Device* device, uint32 capacity) noexcept
        {
            return new TextureDescriptorHeap{
                .state = device,
                .capacity = capacity,
                .compatible_views = new VkImageView[capacity]{},
            };
        }

        // Null views, of slots never written, are ignored.
        void destroy_texture_descriptor_heap(TextureDescriptorHeap* heap) noexcept
        {
            for (uint32 index = 0; index < heap->capacity; ++index)
                vkDestroyImageView(heap->state->device, heap->compatible_views[index], nullptr);
            delete[] heap->compatible_views;
            delete heap;
        }

        // The slot's last use has completed, as the API requires before a slot is written again, so its old view can go.
        void write_texture_descriptor(TextureDescriptorHeap* heap, uint32 index, TextureDescriptorType type, const VkImageViewCreateInfo& view_info) noexcept
        {
            Device& device = *heap->state;
            VkImageView& view = heap->compatible_views[index];
            vkDestroyImageView(device.device, view, nullptr);
            assert_vk(vkCreateImageView(device.device, &view_info, nullptr, &view));
            const bool is_sampled = type == TextureDescriptorType::sampled;
            const VkDescriptorImageInfo image_info{
                .imageView = view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = device.compatible_descriptors.set,
                .dstBinding = is_sampled ? sampled_texture_binding : storage_texture_binding,
                .dstArrayElement = index,
                .descriptorCount = 1,
                .descriptorType = is_sampled ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &image_info,
            };
            vkUpdateDescriptorSets(device.device, 1, &write, 0, nullptr);
        }

        SamplerDescriptorHeap* create_sampler_descriptor_heap(Device* device, uint32 capacity) noexcept
        {
            return new SamplerDescriptorHeap{
                .state = device,
                .capacity = capacity,
                .compatible_samplers = new VkSampler[capacity]{},
            };
        }

        // Null samplers, of slots never written, are ignored.
        void destroy_sampler_descriptor_heap(SamplerDescriptorHeap* heap) noexcept
        {
            for (uint32 index = 0; index < heap->capacity; ++index)
                vkDestroySampler(heap->state->device, heap->compatible_samplers[index], nullptr);
            delete[] heap->compatible_samplers;
            delete heap;
        }

        // As for a texture descriptor, the slot's old sampler can go.
        void write_sampler_descriptor(SamplerDescriptorHeap* heap, uint32 index, const VkSamplerCreateInfo& sampler_info) noexcept
        {
            Device& device = *heap->state;
            VkSampler& sampler = heap->compatible_samplers[index];
            vkDestroySampler(device.device, sampler, nullptr);
            assert_vk(vkCreateSampler(device.device, &sampler_info, nullptr, &sampler));
            const VkDescriptorImageInfo sampler_descriptor{
                .sampler = sampler,
            };
            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = device.compatible_descriptors.set,
                .dstBinding = sampler_binding,
                .dstArrayElement = index,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                .pImageInfo = &sampler_descriptor,
            };
            vkUpdateDescriptorSets(device.device, 1, &write, 0, nullptr);
        }

        // A command buffer on a general queue binds the set for graphics too.
        void bind_descriptors(CommandBuffer* commands) noexcept
        {
            const detail::CompatibleDescriptors& descriptors = commands->state->compatible_descriptors;
            vkCmdBindDescriptorSets(commands->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, descriptors.pipeline_layout, 0, 1, &descriptors.set, 0, nullptr);
            if (commands->graphics)
                vkCmdBindDescriptorSets(
                    commands->command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    descriptors.pipeline_layout,
                    0,
                    1,
                    &descriptors.set,
                    0,
                    nullptr
                );
        }

        void push_root(CommandBuffer* commands, const void* root) noexcept
        {
            vkCmdPushConstants(commands->command_buffer, commands->state->compatible_descriptors.pipeline_layout, root_stages, 0, sizeof(root), &root);
        }

        void bind_index_buffer(CommandBuffer* commands, GpuRange indices, IndexType type) noexcept
        {
            const BufferLocation index_buffer = find_heap_buffer(*commands->state, indices.gpu);
            vkCmdBindIndexBuffer(commands->command_buffer, index_buffer.buffer, index_buffer.offset, static_cast<VkIndexType>(type));
        }

        void draw_indexed(
            CommandBuffer* commands,
            GpuRange indices,
            IndexType type,
            uint32 index_count,
            uint32 instance_count,
            uint32 first_index,
            int32 vertex_offset,
            uint32 first_instance
        ) noexcept
        {
            bind_index_buffer(commands, indices, type);
            vkCmdDrawIndexed(commands->command_buffer, index_count, instance_count, first_index, vertex_offset, first_instance);
        }

        void draw_indirect(CommandBuffer* commands, GpuRange arguments, uint32 draw_count, uint32 stride) noexcept
        {
            const BufferLocation argument_buffer = find_heap_buffer(*commands->state, arguments.gpu);
            const uint32 argument_stride = stride == 0 ? sizeof(VkDrawIndirectCommand) : stride;
            vkCmdDrawIndirect(commands->command_buffer, argument_buffer.buffer, argument_buffer.offset, draw_count, argument_stride);
        }

        void draw_indexed_indirect(CommandBuffer* commands, GpuRange indices, IndexType type, GpuRange arguments, uint32 draw_count, uint32 stride) noexcept
        {
            bind_index_buffer(commands, indices, type);
            const BufferLocation argument_buffer = find_heap_buffer(*commands->state, arguments.gpu);
            const uint32 argument_stride = stride == 0 ? sizeof(VkDrawIndexedIndirectCommand) : stride;
            vkCmdDrawIndexedIndirect(commands->command_buffer, argument_buffer.buffer, argument_buffer.offset, draw_count, argument_stride);
        }

        void dispatch_indirect(CommandBuffer* commands, GpuRange arguments) noexcept
        {
            const BufferLocation argument_buffer = find_heap_buffer(*commands->state, arguments.gpu);
            vkCmdDispatchIndirect(commands->command_buffer, argument_buffer.buffer, argument_buffer.offset);
        }

        void copy_memory(CommandBuffer* commands, GpuRange source, GpuRange destination) noexcept
        {
            const BufferLocation from = find_heap_buffer(*commands->state, source.gpu);
            const BufferLocation to = find_heap_buffer(*commands->state, destination.gpu);
            const VkBufferCopy region{
                .srcOffset = from.offset,
                .dstOffset = to.offset,
                .size = source.size,
            };
            vkCmdCopyBuffer(commands->command_buffer, from.buffer, to.buffer, 1, &region);
        }

        // The region make_texture_copy_region describes by address, at this offset into the buffer that holds the address.
        VkBufferImageCopy buffer_image_copy(VkDeviceSize buffer_offset, const VkDeviceMemoryImageCopyKHR& region) noexcept
        {
            return {
                .bufferOffset = buffer_offset,
                .bufferRowLength = region.addressRowLength,
                .bufferImageHeight = region.addressImageHeight,
                .imageSubresource = region.imageSubresource,
                .imageOffset = region.imageOffset,
                .imageExtent = region.imageExtent,
            };
        }

        void copy_memory_to_texture(CommandBuffer* commands, GpuRange source, Texture* destination, const VkDeviceMemoryImageCopyKHR& region) noexcept
        {
            const BufferLocation memory = find_heap_buffer(*commands->state, source.gpu);
            const VkBufferImageCopy buffer_region = buffer_image_copy(memory.offset, region);
            vkCmdCopyBufferToImage(commands->command_buffer, memory.buffer, destination->image, VK_IMAGE_LAYOUT_GENERAL, 1, &buffer_region);
        }

        void copy_texture_to_memory(CommandBuffer* commands, Texture* source, GpuRange destination, const VkDeviceMemoryImageCopyKHR& region) noexcept
        {
            const BufferLocation memory = find_heap_buffer(*commands->state, destination.gpu);
            const VkBufferImageCopy buffer_region = buffer_image_copy(memory.offset, region);
            vkCmdCopyImageToBuffer(commands->command_buffer, source->image, VK_IMAGE_LAYOUT_GENERAL, memory.buffer, 1, &buffer_region);
        }

        // The descriptor set is written on the host, so the descriptor heap reads the full tier orders have nothing to order
        // here, and a device without mesh shaders has no task or mesh stage to name.
        void barrier(VkCommandBuffer command_buffer, Stage before, Access before_access, Stage after, Access after_access) noexcept
        {
            constexpr uint64 mesh_stages = static_cast<uint64>(Stage::task) | static_cast<uint64>(Stage::mesh);
            constexpr uint64 descriptor_reads = static_cast<uint64>(Access::descriptor_read);
            const Stage stages_before = static_cast<Stage>(static_cast<uint64>(before) & ~mesh_stages);
            const Stage stages_after = static_cast<Stage>(static_cast<uint64>(after) & ~mesh_stages);
            const Access accesses_before = static_cast<Access>(static_cast<uint64>(before_access) & ~descriptor_reads);
            const Access accesses_after = static_cast<Access>(static_cast<uint64>(after_access) & ~descriptor_reads);
            record_barrier(command_buffer, stages_before, accesses_before, stages_after, accesses_after);
        }

        // Presentation without present fences, on a device without swapchain_maintenance1: nothing says when a present is
        // done with its semaphore or its swapchain, so it presents the classic way. A context's acquire semaphore is free again
        // once the submit that waited on it completes on the presentation retirement timeline (Device::wait_present_context).
        // The semaphore a present waits on is its swapchain image's, because the image coming back from an acquire shows that
        // its last present is done with it. An old swapchain is destroyed once the presenting queue is idle.

        // One for each image a swapchain can have. They outlive every recreate, which leaves the presenting queue idle.
        Error create_image_semaphores(Swapchain& swapchain) noexcept
        {
            const VkSemaphoreCreateInfo semaphore_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            };
            for (VkSemaphore& rendered : swapchain.rendered)
            {
                const VkResult result = vkCreateSemaphore(swapchain.state->device, &semaphore_info, nullptr, &rendered);
                const Error error = error_from_vk(result, "vkCreateSemaphore (render)");
                if (error != Error::none)
                    return error;
            }
            return Error::none;
        }

        // Null semaphores, where creation stopped early, are ignored.
        void destroy_image_semaphores(const Swapchain& swapchain) noexcept
        {
            for (VkSemaphore rendered : swapchain.rendered)
                vkDestroySemaphore(swapchain.state->device, rendered, nullptr);
        }

        // Before an old swapchain is retired. The presenting submits retire with the queue, so the old swapchain and its
        // views are destroyed at once.
        void wait_for_presents(Device& device) noexcept
        {
            assert_vk(vkQueueWaitIdle(device.queues[0].queue));
            device.poll_presentation_retirement();
        }

        void submit_and_present(Device& device, const SubmitDesc& desc) noexcept
        {
            Swapchain& swapchain = *device.swapchain;
            detail::PresentContext& present_context = *swapchain.present_context;
            const VkSemaphore rendered = swapchain.rendered[swapchain.image_index];
            submit_commands(&device, desc, 0, present_context.acquired, rendered);
            present_context.acquire_retirement = device.presentation_retirement_value;
            swapchain.transition_commands->swapchain = nullptr;
            swapchain.transition_commands = nullptr;
            swapchain.initialized[swapchain.image_index] = true;

            const VkPresentInfoKHR present_info{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &rendered,
                .swapchainCount = 1,
                .pSwapchains = &swapchain.handle,
                .pImageIndices = &swapchain.image_index,
            };
            const VkResult result = vkQueuePresentKHR(device.queues[0].queue, &present_info);
            swapchain.present_context = nullptr;
            swapchain.acquired = false;
            device.acquired_swapchain = nullptr;
            device.next_present_context = (device.next_present_context + 1) % device.present_context_count;
            if (result == VK_ERROR_OUT_OF_DATE_KHR)
                swapchain.recreate_required = true;
            else if (result == VK_SUBOPTIMAL_KHR)
                swapchain.recreate_required = swapchain.recreate_required || swapchain_surface_configuration_changed(swapchain);
            else if (result != VK_SUCCESS)
                abort_vk_failure(result, "vkQueuePresentKHR");
        }

    } // namespace compatible
} // namespace
