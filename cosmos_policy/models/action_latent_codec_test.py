import torch

from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence
from cosmos_policy.models.action_latent_codec import ActionLatentCodec
from cosmos_policy.models.policy_text2world_model import replace_latent_with_action_chunk


def test_action_latent_codec_encodes_and_decodes_frame_shape():
    codec = ActionLatentCodec(
        chunk_size=32,
        action_dim=7,
        latent_channels=16,
        latent_height=28,
        latent_width=28,
        bottleneck_dim=64,
        hidden_dim=128,
    )
    actions = torch.randn(3, 32, 7)

    encoded = codec.encode_frame(actions)
    decoded = codec.decode_frame(encoded)

    assert encoded.shape == (3, 16, 28, 28)
    assert decoded.shape == (3, 32, 7)


def test_action_latent_codec_allows_gradients_through_reconstruction():
    codec = ActionLatentCodec(
        chunk_size=32,
        action_dim=7,
        latent_channels=16,
        latent_height=28,
        latent_width=28,
        bottleneck_dim=64,
        hidden_dim=128,
    )
    actions = torch.randn(2, 32, 7)

    decoded = codec(actions)
    loss = torch.nn.functional.mse_loss(decoded, actions)
    loss.backward()

    assert codec.encoder[0].weight.grad is not None
    assert codec.decoder[-1].weight.grad is not None


def test_repeat_fill_action_roundtrip_without_codec():
    x0 = torch.zeros(2, 16, 3, 4, 4)
    actions = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    action_indices = torch.tensor([1, 2])

    latents = replace_latent_with_action_chunk(x0, actions, action_indices)
    extracted = extract_action_chunk_from_latent_sequence(latents, (4, 3), action_indices)

    assert torch.allclose(extracted, actions)


def test_extract_action_chunk_decodes_with_codec():
    codec = ActionLatentCodec(
        chunk_size=32,
        action_dim=7,
        latent_channels=16,
        latent_height=28,
        latent_width=28,
        bottleneck_dim=64,
        hidden_dim=128,
    )
    actions = torch.randn(2, 32, 7)
    action_indices = torch.tensor([0, 2])
    output_latent = torch.zeros(2, 16, 3, 28, 28)
    batch_indices = torch.arange(2)
    output_latent[batch_indices, :, action_indices, :, :] = codec.encode_frame(actions)

    extracted = extract_action_chunk_from_latent_sequence(
        output_latent,
        (32, 7),
        action_indices,
        action_latent_codec=codec,
    )

    assert extracted.shape == actions.shape
