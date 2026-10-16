// The compatible tier's descriptor set, which shader.slang imports. Being a module, it is declared once however many of a
// shader's modules include shader.slang.

// Slot i of each binding is descriptor heap index i.
[[vk::binding(0, 0)]] __DynamicResource<__DynamicResourceKind.General> gpu_sampled_textures[];
[[vk::binding(1, 0)]] __DynamicResource<__DynamicResourceKind.General> gpu_storage_textures[];
[[vk::binding(2, 0)]] __DynamicResource<__DynamicResourceKind.Sampler> gpu_samplers[];

[ForceInline]
T gpu_texture<T : IOpaqueDescriptor>(uint index)
{
    if (T.descriptorAccess == DescriptorAccess.Read) return gpu_sampled_textures[index].asOpaqueDescriptor<T>();
    return gpu_storage_textures[index].asOpaqueDescriptor<T>();
}

[ForceInline]
T gpu_sampler<T : IOpaqueDescriptor = SamplerState>(uint index)
{
    return gpu_samplers[index].asOpaqueDescriptor<T>();
}
