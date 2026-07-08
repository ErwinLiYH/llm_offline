import re
import unittest
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from data.pointmaze import dataset as pointmaze_dataset
from model.continuous_action import (
    CONTINUOUS_ACTION_DECODER_FILENAME,
    CONTINUOUS_POLICY_GAUSSIAN,
    CONTINUOUS_POLICY_STUDENT_T,
    ContinuousActionDecoder,
    GaussianActionOutput,
    attach_continuous_action_decoder,
    build_parallel_action_attention_mask,
    load_continuous_action_decoder,
    resolve_continuous_policy_type,
    resolve_action_head_num_blocks,
    resolve_action_query_len,
    resolve_gaussian_log_std_bounds,
    resolve_gaussian_log_std_init,
    resolve_student_t_df,
    save_continuous_action_decoder,
    squashed_gaussian_negative_log_likelihood,
    student_t_negative_log_likelihood,
)
from model.mtp_bin import (
    MTP_BIN_DECODER_FILENAME,
    MTPBinOutput,
    attach_mtp_bin_decoder,
    build_mtp_bin_attention_mask,
    load_mtp_bin_decoder,
    mtp_bin_action_loss,
    resolve_mtp_k,
    resolve_mtp_quadratic_decoding,
    save_mtp_bin_decoder,
)
from utils.action_bins import (
    action_bin_equivalent_l1,
    get_action_bin_codec,
    get_action_token_mode,
    uses_action_bins,
    uses_continuous_actions,
    uses_student_t_continuous_actions,
)
from utils.eval_rollout import build_action_rollout_context, generate_valid_action


class DummyCausalModel(nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.last_attention_mask = None
        self.last_inputs_embeds = None

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids=None,
        output_hidden_states,
        return_dict,
        use_cache,
    ):
        self.last_attention_mask = attention_mask
        self.last_inputs_embeds = inputs_embeds
        return SimpleNamespace(logits=self.lm_head(inputs_embeds), hidden_states=(inputs_embeds,))


class CaptureActionHead(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.dtype_anchor = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.action_dim = action_dim
        self.last_action_hidden = None

    def forward(self, action_hidden):
        self.last_action_hidden = action_hidden.detach().clone()
        return torch.zeros((action_hidden.shape[0], self.action_dim), device=action_hidden.device)


class DummyEncoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    @property
    def attention_mask(self):
        return self["attention_mask"]


class DummyTokenizer:
    def __init__(self, vocab_size: int = 16):
        self.name_or_path = "dummy-tokenizer"
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = None
        self.unk_token_id = None
        self.chat_template = "dummy"
        self.additional_special_tokens = []
        self.id_to_token = {idx: f"<tok{idx}>" for idx in range(vocab_size)}
        self.token_to_id = {token: idx for idx, token in self.id_to_token.items()}
        self.all_special_ids = [0, 1]

    def __len__(self):
        return len(self.id_to_token)

    def add_special_tokens(self, payload):
        added = 0
        for token in payload.get("additional_special_tokens", []):
            if token not in self.token_to_id:
                token_id = len(self.id_to_token)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                self.all_special_ids.append(token_id)
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        return added

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.token_to_id.get(tokens, -1)
        return [self.token_to_id.get(token, -1) for token in tokens]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            pieces.append(self.id_to_token[token_id])
        return "".join(pieces)

    def _encode_text(self, text: str) -> list[int]:
        pattern = re.compile(r"<act_\d+>|<tok\d+>")
        ids = []
        pos = 0
        while pos < len(text):
            match = pattern.match(text, pos)
            if match:
                token = match.group(0)
                ids.append(self.token_to_id.get(token, 2))
                pos = match.end()
                continue
            if not text[pos].isspace():
                ids.append(2)
            pos += 1
        return ids

    def __call__(
        self,
        *,
        text,
        add_special_tokens=False,
        max_length=None,
        truncation=False,
        return_tensors=None,
    ):
        ids = self._encode_text(text)
        if max_length is not None and truncation:
            ids = ids[: int(max_length)]
        attention = [1] * len(ids)
        if return_tensors == "pt":
            return DummyEncoding(
                {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.tensor([attention], dtype=torch.long),
                }
            )
        return DummyEncoding({"input_ids": ids, "attention_mask": attention})

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = ""
        for message in messages:
            role = message["role"].upper()
            text += f"{role}: {message['content']}\n"
        if add_generation_prompt:
            text += "ASSISTANT:"
        return text


class DummyFormatter:
    def format_action(self, action):
        return ",".join(f"{float(value):.2f}" for value in action)

    def parse_action(self, text):
        raise AssertionError("mtp_bin should not use formatter.parse_action")

    def validate_action(self, action):
        arr = np.asarray(action, dtype=np.float32)
        return arr.shape == (2,) and bool(np.all(arr >= -1.0) and np.all(arr <= 1.0))


class IdentitySamplerHead(nn.Module):
    def forward(self, action_query_hidden, previous_token_embedding):
        return action_query_hidden


class DummyMTPBinDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampler_head = IdentitySamplerHead()


class DummyMTPBinModel(nn.Module):
    def __init__(self, *, vocab_size: int, bin_token_ids: list[int]):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)
        self.lm_head = nn.Linear(vocab_size, vocab_size, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(vocab_size))
            self.lm_head.weight.copy_(torch.eye(vocab_size))
        self.bin_token_ids = list(bin_token_ids)
        self.mtp_bin_decoder = DummyMTPBinDecoder()
        self.forward_calls = 0
        self.last_attention_mask = None

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids=None,
        action_query_mask=None,
        action_query_offsets=None,
        action_query_source_positions=None,
        action_query_prev_token_ids=None,
        mtp_bin=False,
    ):
        if not mtp_bin:
            raise AssertionError("DummyMTPBinModel expects mtp_bin=True")
        self.forward_calls += 1
        self.last_attention_mask = attention_mask
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.embed.num_embeddings,
            device=input_ids.device,
        )
        hidden = torch.zeros_like(logits)
        for row in range(input_ids.shape[0]):
            query_positions = torch.nonzero(action_query_mask[row], as_tuple=False).flatten().tolist()
            non_query_positions = torch.nonzero(
                attention_mask[row].bool() & ~action_query_mask[row],
                as_tuple=False,
            ).flatten().tolist()
            source_pos = non_query_positions[-1]
            source_token_id = int(input_ids[row, source_pos].item())
            first_bin_token_id = self.bin_token_ids[2]
            next_bin = 0 if source_token_id == first_bin_token_id else 2
            logits[row, source_pos, self.bin_token_ids[next_bin]] = 10.0
            for position in query_positions:
                hidden[row, position, self.bin_token_ids[0]] = 10.0
        return MTPBinOutput(
            logits=logits,
            hidden_states=hidden,
            sampler_logits=torch.zeros((0, self.embed.num_embeddings), device=input_ids.device),
            sampler_query_mask=action_query_mask,
        )

    def generate(self, *args, **kwargs):
        raise AssertionError("mtp_bin eval must not call generate()")


class ContinuousActionTest(unittest.TestCase):
    def test_action_mode_helpers_accept_parallel_l1(self):
        config = {"action_token_mode": "parallel_l1"}

        self.assertEqual(get_action_token_mode(config), "parallel_l1")
        self.assertEqual(resolve_continuous_policy_type(config), "deterministic")
        self.assertTrue(uses_continuous_actions(config))
        self.assertFalse(uses_action_bins(config))

    def test_action_mode_helpers_accept_parallel_gaussian(self):
        config = {"action_token_mode": "parallel_gaussian"}

        self.assertEqual(get_action_token_mode(config), "parallel_gaussian")
        self.assertEqual(resolve_continuous_policy_type(config), CONTINUOUS_POLICY_GAUSSIAN)
        self.assertTrue(uses_continuous_actions(config))
        self.assertFalse(uses_action_bins(config))

    def test_action_mode_helpers_accept_parallel_t(self):
        config = {"action_token_mode": "parallel_t"}

        self.assertEqual(get_action_token_mode(config), "parallel_t")
        self.assertEqual(resolve_continuous_policy_type(config), CONTINUOUS_POLICY_STUDENT_T)
        self.assertTrue(uses_continuous_actions(config))
        self.assertTrue(uses_student_t_continuous_actions(config))
        self.assertFalse(uses_action_bins(config))

    def test_parallel_action_attention_mask_policy(self):
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

        additive_mask = build_parallel_action_attention_mask(
            attention_mask,
            query_len=2,
            dtype=torch.float32,
        )
        allowed = additive_mask[0, 0] == 0

        self.assertEqual(tuple(additive_mask.shape), (1, 1, 5, 5))
        self.assertEqual(allowed[0].tolist(), [True, False, False, False, False])
        self.assertEqual(allowed[1].tolist(), [True, True, False, False, False])
        self.assertEqual(allowed[2].tolist(), [True, True, True, False, False])
        self.assertEqual(allowed[3].tolist(), [True, True, True, True, True])
        self.assertEqual(allowed[4].tolist(), [True, True, True, True, True])

    def test_resolve_action_query_len_defaults_to_action_dim(self):
        self.assertEqual(resolve_action_query_len(2), 2)
        self.assertEqual(resolve_action_query_len(2, 8), 8)
        with self.assertRaises(ValueError):
            resolve_action_query_len(2, 0)

    def test_resolve_action_head_num_blocks_defaults_to_two(self):
        self.assertEqual(resolve_action_head_num_blocks(), 2)
        self.assertEqual(resolve_action_head_num_blocks(8), 8)
        with self.assertRaises(ValueError):
            resolve_action_head_num_blocks(0)

    def test_decoder_appends_learned_action_queries_and_uses_parallel_attention_mask(self):
        for action_dim, action_query_len in ((2, 2), (2, 8)):
            with self.subTest(action_dim=action_dim, action_query_len=action_query_len):
                model = DummyCausalModel(hidden_size=12)
                decoder = ContinuousActionDecoder(
                    action_dim=action_dim,
                    hidden_size=12,
                    action_query_len=action_query_len,
                )
                prompt_ids = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)
                prompt_attention = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
                capture_head = CaptureActionHead(action_dim=action_dim)
                decoder.action_head = capture_head

                actions = decoder(
                    model,
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention,
                )

                self.assertEqual(tuple(actions.shape), (2, action_dim))
                self.assertEqual(
                    tuple(model.last_attention_mask.shape),
                    (
                        prompt_ids.shape[0],
                        1,
                        prompt_ids.shape[1] + action_query_len,
                        prompt_ids.shape[1] + action_query_len,
                    ),
                )
                prompt_embeds = model.get_input_embeddings()(prompt_ids)
                self.assertTrue(torch.allclose(model.last_inputs_embeds[:, : prompt_ids.shape[1], :], prompt_embeds))
                expected_action_hidden = decoder.action_queries.unsqueeze(0).expand(prompt_ids.shape[0], -1, -1)
                self.assertTrue(torch.allclose(capture_head.last_action_hidden, expected_action_hidden))

    def test_decoder_has_learned_action_query_parameters(self):
        decoder = ContinuousActionDecoder(action_dim=2, hidden_size=10)

        parameter_dict = dict(decoder.named_parameters())

        self.assertIn("action_queries", parameter_dict)
        self.assertEqual(tuple(parameter_dict["action_queries"].shape), (2, 10))
        self.assertTrue(parameter_dict["action_queries"].requires_grad)

        wider_decoder = ContinuousActionDecoder(action_dim=2, hidden_size=10, action_query_len=8)
        wider_parameter_dict = dict(wider_decoder.named_parameters())
        self.assertEqual(tuple(wider_parameter_dict["action_queries"].shape), (8, 10))
        self.assertEqual(wider_decoder.action_head.input_norm.normalized_shape, (80,))

        deeper_decoder = ContinuousActionDecoder(
            action_dim=2,
            hidden_size=10,
            action_head_num_blocks=4,
        )
        self.assertEqual(deeper_decoder.action_head_num_blocks, 4)
        self.assertEqual(len(deeper_decoder.action_head.blocks), 4)

    def test_gaussian_decoder_outputs_squashed_mean_and_state_independent_std(self):
        model = DummyCausalModel(hidden_size=12)
        decoder = ContinuousActionDecoder(
            action_dim=2,
            hidden_size=12,
            action_query_len=4,
            policy_type=CONTINUOUS_POLICY_GAUSSIAN,
            gaussian_log_std_min=-3.0,
            gaussian_log_std_max=0.5,
            gaussian_log_std_init=-1.25,
        )
        prompt_ids = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)
        prompt_attention = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)

        output = decoder(model, input_ids=prompt_ids, attention_mask=prompt_attention)

        self.assertIsInstance(output, GaussianActionOutput)
        self.assertEqual(tuple(output.mean.shape), (2, 2))
        self.assertEqual(tuple(output.latent_mean.shape), (2, 2))
        self.assertEqual(tuple(output.log_std.shape), (2, 2))
        self.assertEqual(tuple(output.std.shape), (2, 2))
        self.assertEqual(decoder.action_head.output_proj.out_features, 2)
        self.assertEqual(tuple(decoder.gaussian_log_std.shape), (2,))
        self.assertTrue(torch.all(output.mean <= 1.0))
        self.assertTrue(torch.all(output.mean >= -1.0))
        self.assertTrue(torch.allclose(output.mean, torch.tanh(output.latent_mean)))
        self.assertTrue(torch.all(output.log_std <= 0.5))
        self.assertTrue(torch.all(output.log_std >= -3.0))
        self.assertTrue(torch.allclose(output.log_std[0], output.log_std[1]))
        self.assertTrue(torch.allclose(output.log_std[0], torch.full((2,), -1.25)))
        self.assertTrue(torch.all(output.std > 0.0))
        self.assertTrue(torch.allclose(output.log_scale, output.log_std))
        self.assertTrue(torch.allclose(output.scale, output.std))

    def test_student_t_decoder_outputs_bounded_mean_and_scale(self):
        model = DummyCausalModel(hidden_size=12)
        decoder = ContinuousActionDecoder(
            action_dim=2,
            hidden_size=12,
            action_query_len=4,
            policy_type=CONTINUOUS_POLICY_STUDENT_T,
            gaussian_log_std_min=-3.0,
            gaussian_log_std_max=0.5,
        )
        prompt_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
        prompt_attention = torch.tensor([[1, 1, 1]], dtype=torch.long)

        output = decoder(model, input_ids=prompt_ids, attention_mask=prompt_attention)

        self.assertIsInstance(output, GaussianActionOutput)
        self.assertEqual(tuple(output.mean.shape), (1, 2))
        self.assertEqual(tuple(output.log_scale.shape), (1, 2))
        self.assertEqual(tuple(output.scale.shape), (1, 2))
        self.assertTrue(torch.all(output.mean <= 1.0))
        self.assertTrue(torch.all(output.mean >= -1.0))
        self.assertTrue(torch.all(output.log_scale <= 0.5))
        self.assertTrue(torch.all(output.log_scale >= -3.0))
        self.assertTrue(torch.all(output.scale > 0.0))

    def test_resolve_gaussian_log_std_bounds(self):
        self.assertEqual(resolve_gaussian_log_std_bounds({}), (-5.0, 1.0))
        self.assertEqual(
            resolve_gaussian_log_std_bounds(
                {"gaussian_log_std_min": None, "gaussian_log_std_max": None}
            ),
            (-5.0, 1.0),
        )
        self.assertEqual(
            resolve_gaussian_log_std_bounds(
                {"gaussian_log_std_min": -4, "gaussian_log_std_max": 0}
            ),
            (-4.0, 0.0),
        )
        with self.assertRaises(ValueError):
            resolve_gaussian_log_std_bounds(
                {"gaussian_log_std_min": 0, "gaussian_log_std_max": 0}
            )

    def test_resolve_gaussian_log_std_init(self):
        self.assertEqual(resolve_gaussian_log_std_init({}), -1.0)
        self.assertEqual(resolve_gaussian_log_std_init({"gaussian_log_std_init": None}), -1.0)
        self.assertEqual(resolve_gaussian_log_std_init({"gaussian_log_std_init": -2}), -2.0)

    def test_resolve_student_t_df(self):
        self.assertEqual(resolve_student_t_df({}), 3.0)
        self.assertEqual(resolve_student_t_df({"student_t_df": None}), 3.0)
        self.assertEqual(resolve_student_t_df({"student_t_df": 5}), 5.0)
        with self.assertRaises(ValueError):
            resolve_student_t_df({"student_t_df": 0})

    def test_student_t_negative_log_likelihood_matches_torch_distribution(self):
        target = torch.tensor([[0.0, 0.5]], dtype=torch.float32)
        mean = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        log_scale = torch.zeros_like(target)
        df = 3.0

        nll = student_t_negative_log_likelihood(target, mean, log_scale, df)
        expected = -torch.distributions.StudentT(df, loc=mean, scale=torch.ones_like(target)).log_prob(
            target
            )

        self.assertTrue(torch.allclose(nll, expected, atol=1e-6))

    def test_squashed_gaussian_negative_log_likelihood_matches_change_of_variables(self):
        target = torch.tensor([[0.0, 0.5]], dtype=torch.float32)
        latent_mean = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
        log_std = torch.tensor([[-0.4, 0.2]], dtype=torch.float32)
        eps = 1e-6

        nll = squashed_gaussian_negative_log_likelihood(
            target,
            latent_mean,
            log_std,
            eps=eps,
        )
        clipped = target.clamp(-1.0 + eps, 1.0 - eps)
        latent_target = torch.atanh(clipped)
        expected = -torch.distributions.Normal(
            latent_mean,
            torch.exp(log_std),
        ).log_prob(latent_target) + torch.log1p(-clipped.square() + eps)

        self.assertTrue(torch.allclose(nll, expected, atol=1e-6))

    def test_attach_save_and_load_decoder_sidecar(self):
        model = DummyCausalModel(hidden_size=10)
        decoder = attach_continuous_action_decoder(
            model,
            action_dim=2,
            action_query_len=8,
            action_head_num_blocks=4,
        )
        with torch.no_grad():
            decoder.action_queries[0, 0] = 0.5
            decoder.action_head.output_proj.bias.fill_(0.125)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_continuous_action_decoder(model, tmpdir)
            payload = torch.load(f"{tmpdir}/{CONTINUOUS_ACTION_DECODER_FILENAME}", map_location="cpu")
            self.assertEqual(payload["action_query_len"], 8)
            self.assertEqual(payload["action_head_num_blocks"], 4)
            self.assertEqual(payload["policy_type"], "deterministic")

            reloaded = DummyCausalModel(hidden_size=10)
            load_continuous_action_decoder(
                reloaded,
                tmpdir,
                expected_action_dim=2,
                expected_action_query_len=8,
                expected_action_head_num_blocks=4,
                expected_policy_type="deterministic",
            )

            self.assertTrue(hasattr(reloaded, "continuous_action_decoder"))
            self.assertEqual(reloaded.continuous_action_decoder.action_dim, 2)
            self.assertEqual(reloaded.continuous_action_decoder.action_query_len, 8)
            self.assertEqual(reloaded.continuous_action_decoder.action_head_num_blocks, 4)
            self.assertTrue(
                torch.allclose(
                    reloaded.continuous_action_decoder.action_queries,
                    decoder.action_queries,
                )
            )
            self.assertTrue(
                torch.allclose(
                    reloaded.continuous_action_decoder.action_head.output_proj.bias,
                    decoder.action_head.output_proj.bias,
                )
            )

    def test_attach_save_and_load_gaussian_decoder_sidecar(self):
        model = DummyCausalModel(hidden_size=10)
        decoder = attach_continuous_action_decoder(
            model,
            action_dim=2,
            action_query_len=4,
            action_head_num_blocks=3,
            policy_type=CONTINUOUS_POLICY_GAUSSIAN,
            gaussian_log_std_min=-4.0,
            gaussian_log_std_max=0.25,
            gaussian_log_std_init=-1.5,
        )
        with torch.no_grad():
            decoder.gaussian_log_std.fill_(-0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_continuous_action_decoder(model, tmpdir)
            payload = torch.load(f"{tmpdir}/{CONTINUOUS_ACTION_DECODER_FILENAME}", map_location="cpu")
            self.assertEqual(payload["policy_type"], CONTINUOUS_POLICY_GAUSSIAN)
            self.assertEqual(payload["gaussian_log_std_min"], -4.0)
            self.assertEqual(payload["gaussian_log_std_max"], 0.25)
            self.assertEqual(payload["gaussian_log_std_init"], -1.5)

            reloaded = DummyCausalModel(hidden_size=10)
            load_continuous_action_decoder(
                reloaded,
                tmpdir,
                expected_action_dim=2,
                expected_action_query_len=4,
                expected_action_head_num_blocks=3,
                expected_policy_type=CONTINUOUS_POLICY_GAUSSIAN,
                expected_gaussian_log_std_min=-4.0,
                expected_gaussian_log_std_max=0.25,
            )

            self.assertEqual(reloaded.continuous_action_decoder.policy_type, CONTINUOUS_POLICY_GAUSSIAN)
            self.assertEqual(reloaded.continuous_action_decoder.gaussian_log_std_min, -4.0)
            self.assertEqual(reloaded.continuous_action_decoder.gaussian_log_std_max, 0.25)
            self.assertEqual(
                reloaded.continuous_action_decoder.action_head.output_proj.out_features,
                2,
            )
            self.assertTrue(
                torch.allclose(
                    reloaded.continuous_action_decoder.gaussian_log_std,
                    torch.full((2,), -0.5),
                )
            )

    def test_attach_save_and_load_student_t_decoder_sidecar(self):
        model = DummyCausalModel(hidden_size=10)
        attach_continuous_action_decoder(
            model,
            action_dim=2,
            action_query_len=4,
            action_head_num_blocks=3,
            policy_type=CONTINUOUS_POLICY_STUDENT_T,
            gaussian_log_std_min=-4.0,
            gaussian_log_std_max=0.25,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_continuous_action_decoder(model, tmpdir)
            payload = torch.load(f"{tmpdir}/{CONTINUOUS_ACTION_DECODER_FILENAME}", map_location="cpu")
            self.assertEqual(payload["policy_type"], CONTINUOUS_POLICY_STUDENT_T)
            self.assertEqual(payload["gaussian_log_std_min"], -4.0)
            self.assertEqual(payload["gaussian_log_std_max"], 0.25)

            reloaded = DummyCausalModel(hidden_size=10)
            load_continuous_action_decoder(
                reloaded,
                tmpdir,
                expected_action_dim=2,
                expected_action_query_len=4,
                expected_action_head_num_blocks=3,
                expected_policy_type=CONTINUOUS_POLICY_STUDENT_T,
                expected_gaussian_log_std_min=-4.0,
                expected_gaussian_log_std_max=0.25,
            )

            self.assertEqual(reloaded.continuous_action_decoder.policy_type, CONTINUOUS_POLICY_STUDENT_T)
            self.assertEqual(
                reloaded.continuous_action_decoder.action_head.output_proj.out_features,
                4,
            )

    def test_old_sidecar_without_action_query_len_defaults_to_action_dim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decoder = ContinuousActionDecoder(action_dim=2, hidden_size=10)
            torch.save(
                {
                    "action_dim": 2,
                    "hidden_size": 10,
                    "state_dict": decoder.state_dict(),
                },
                f"{tmpdir}/{CONTINUOUS_ACTION_DECODER_FILENAME}",
            )

            reloaded = DummyCausalModel(hidden_size=10)
            load_continuous_action_decoder(reloaded, tmpdir, expected_action_dim=2)

            self.assertEqual(reloaded.continuous_action_decoder.action_dim, 2)
            self.assertEqual(reloaded.continuous_action_decoder.action_query_len, 2)
            self.assertEqual(reloaded.continuous_action_decoder.action_head_num_blocks, 2)
            self.assertEqual(reloaded.continuous_action_decoder.policy_type, "deterministic")

    def test_sidecar_policy_type_mismatch_fails(self):
        model = DummyCausalModel(hidden_size=10)
        attach_continuous_action_decoder(model, action_dim=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_continuous_action_decoder(model, tmpdir)

            with self.assertRaises(ValueError):
                load_continuous_action_decoder(
                    DummyCausalModel(hidden_size=10),
                    tmpdir,
                    expected_action_dim=2,
                    expected_policy_type=CONTINUOUS_POLICY_GAUSSIAN,
                )

    def test_incompatible_missing_action_query_sidecar_fails_state_dict_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dict = ContinuousActionDecoder(action_dim=2, hidden_size=10).state_dict()
            state_dict.pop("action_queries")
            torch.save(
                {
                    "action_dim": 2,
                    "hidden_size": 10,
                    "state_dict": state_dict,
                },
                f"{tmpdir}/{CONTINUOUS_ACTION_DECODER_FILENAME}",
            )

            with self.assertRaises(RuntimeError):
                load_continuous_action_decoder(
                    DummyCausalModel(hidden_size=10),
                    tmpdir,
                    expected_action_dim=2,
                )


class MTPBinTest(unittest.TestCase):
    def test_action_mode_helpers_accept_mtp_bin(self):
        config = {"action_token_mode": "mtp_bin"}

        self.assertEqual(get_action_token_mode(config), "mtp_bin")
        self.assertTrue(uses_action_bins(config))
        self.assertFalse(uses_continuous_actions(config))
        self.assertEqual(resolve_mtp_k(2, None), 1)
        self.assertTrue(resolve_mtp_quadratic_decoding(config))
        self.assertFalse(resolve_mtp_quadratic_decoding({"mtp_quadratic_decoding": "false"}))
        with self.assertRaises(ValueError):
            resolve_mtp_k(2, 2)

    def test_existing_token_codec_reserves_only_action_bins(self):
        tokenizer = DummyTokenizer(vocab_size=12)
        codec = get_action_bin_codec(
            tokenizer,
            {"action_token_mode": "mtp_bin", "action_num_bins": 3, "new_token": False},
            ensure_registered=True,
        )

        self.assertEqual(codec.model_token_ids, (11, 10, 9))
        self.assertEqual(len(set(codec.model_token_ids)), 3)

    def test_new_token_codec_registers_only_action_bins(self):
        tokenizer = DummyTokenizer(vocab_size=8)
        get_action_bin_codec(
            tokenizer,
            {"action_token_mode": "mtp_bin", "action_num_bins": 3, "new_token": True},
            ensure_registered=True,
        )

        self.assertEqual(tokenizer.additional_special_tokens, ["<act_00>", "<act_01>", "<act_02>"])

    def test_mtp_attention_mask_blocks_aqt_from_other_blocks(self):
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)
        action_query_mask = torch.tensor([[False, False, False, True, True, False]])
        source_positions = torch.tensor([[-1, -1, -1, 2, 1, -1]], dtype=torch.long)

        additive_mask = build_mtp_bin_attention_mask(
            attention_mask,
            action_query_mask,
            source_positions,
            dtype=torch.float32,
        )
        allowed = additive_mask[0, 0] == 0

        self.assertEqual(tuple(additive_mask.shape), (1, 1, 6, 6))
        self.assertEqual(allowed[0].tolist(), [True, False, False, False, False, False])
        self.assertEqual(allowed[2].tolist(), [True, True, True, False, False, False])
        self.assertEqual(allowed[3].tolist(), [True, True, True, True, False, False])
        self.assertEqual(allowed[4].tolist(), [True, True, False, False, True, False])
        self.assertEqual(allowed[5].tolist(), [False, False, False, False, False, False])

    def test_mtp_bin_loss_uses_ntp_and_aqt_positions(self):
        logits = torch.zeros((1, 4, 10), dtype=torch.float32)
        hidden = torch.zeros((1, 4, 3), dtype=torch.float32)
        sampler_logits = torch.zeros((1, 10), dtype=torch.float32)
        action_bin_labels = torch.tensor([[0, -1, 2, -1]], dtype=torch.long)
        action_query_mask = torch.tensor([[False, False, True, False]])
        action_query_anchor_positions = torch.tensor([[-1, -1, 1, -1]], dtype=torch.long)
        bin_token_ids = [4, 6, 8]
        logits[0, 0, 4] = 10.0
        logits[0, 2, 8] = 10.0
        sampler_logits[0, 8] = 10.0
        output = MTPBinOutput(
            logits=logits,
            hidden_states=hidden,
            sampler_logits=sampler_logits,
            sampler_query_mask=action_query_mask,
        )

        loss, metrics = mtp_bin_action_loss(
            output,
            action_bin_labels,
            action_query_mask,
            action_query_anchor_positions,
            bin_token_ids,
        )

        self.assertLess(float(loss.item()), 0.002)
        self.assertEqual(metrics["action_tokens"], 2)
        self.assertEqual(metrics["action_query_tokens"], 1)

    def test_action_bin_equivalent_l1_uses_matching_prediction_positions(self):
        bin_token_ids = [4, 6, 8]
        action_bin_labels = torch.tensor([[-1, 0, 2, -1]], dtype=torch.long)

        causal_logits = torch.zeros((1, 4, 10), dtype=torch.float32)
        causal_logits[0, 0, 4] = 10.0
        causal_logits[0, 1, 8] = 10.0
        causal_metric = action_bin_equivalent_l1(
            causal_logits,
            action_bin_labels,
            bin_token_ids,
            num_bins=3,
            low=-1.0,
            high=1.0,
            causal_shift=True,
        )

        parallel_logits = torch.zeros((1, 4, 10), dtype=torch.float32)
        parallel_logits[0, 1, 4] = 10.0
        parallel_logits[0, 2, 8] = 10.0
        parallel_metric = action_bin_equivalent_l1(
            parallel_logits,
            action_bin_labels,
            bin_token_ids,
            num_bins=3,
            low=-1.0,
            high=1.0,
            causal_shift=False,
        )

        self.assertEqual(causal_metric, 0.0)
        self.assertEqual(parallel_metric, 0.0)

    def test_mtp_bin_decoder_save_load_roundtrip(self):
        model = DummyCausalModel(hidden_size=10)
        decoder = attach_mtp_bin_decoder(model, action_dim=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_mtp_bin_decoder(model, tmpdir)
            payload = torch.load(f"{tmpdir}/{MTP_BIN_DECODER_FILENAME}", map_location="cpu")
            self.assertIn("action_query_embeddings.weight", payload["state_dict"])
            reloaded = DummyCausalModel(hidden_size=10)
            load_mtp_bin_decoder(reloaded, tmpdir, expected_action_dim=2)
            self.assertTrue(
                torch.allclose(
                    decoder.action_query_embeddings.weight,
                    reloaded.mtp_bin_decoder.action_query_embeddings.weight,
                )
            )

    def test_mtp_bin_forward_handles_bfloat16_base_with_float32_sampler_head(self):
        model = DummyCausalModel(hidden_size=4, vocab_size=8).to(dtype=torch.bfloat16)
        decoder = attach_mtp_bin_decoder(model, action_dim=2)

        self.assertEqual(decoder.sampler_head.input_norm.weight.dtype, torch.float32)

        output = model(
            input_ids=torch.tensor([[2, 3, 0]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
            action_query_mask=torch.tensor([[False, False, True]], dtype=torch.bool),
            action_query_offsets=torch.tensor([[-1, -1, 0]], dtype=torch.long),
            action_query_source_positions=torch.tensor([[-1, -1, 1]], dtype=torch.long),
            action_query_prev_token_ids=torch.tensor([[0, 0, 3]], dtype=torch.long),
            mtp_bin=True,
        )

        self.assertEqual(tuple(output.sampler_logits.shape), (1, 8))
        self.assertEqual(output.sampler_logits.dtype, torch.bfloat16)

    def test_pointmaze_tokenization_appends_aqt_metadata(self):
        tokenizer = DummyTokenizer(vocab_size=8)
        config = {
            "action_token_mode": "mtp_bin",
            "action_num_bins": 3,
            "action_bin_min": -1.0,
            "action_bin_max": 1.0,
            "new_token": True,
            "action_dim": 2,
            "mtp_k": 1,
            "max_length": 6,
        }
        codec = get_action_bin_codec(tokenizer, config, ensure_registered=True)
        old_tokenizer = pointmaze_dataset._POINTMAZE_WORKER_TOKENIZER
        old_codec = pointmaze_dataset._POINTMAZE_WORKER_ACTION_CODEC
        try:
            pointmaze_dataset._POINTMAZE_WORKER_TOKENIZER = tokenizer
            pointmaze_dataset._POINTMAZE_WORKER_ACTION_CODEC = codec
            sample = pointmaze_dataset._tokenize_pointmaze_sample(
                "prompt",
                "",
                config,
                expected_action_token_ids=codec.token_ids_for_bins([1, 2]),
                expected_action_bin_indices=[1, 2],
            )
        finally:
            pointmaze_dataset._POINTMAZE_WORKER_TOKENIZER = old_tokenizer
            pointmaze_dataset._POINTMAZE_WORKER_ACTION_CODEC = old_codec

        self.assertEqual(len(sample["input_ids"]), 6)
        self.assertEqual(sample["labels"], [-100] * 6)
        self.assertEqual(sample["action_bin_labels"][-3:], [1, 2, 2])
        self.assertEqual(sample["action_query_mask"][-1], True)
        self.assertEqual(sample["action_query_offsets"][-1], 0)
        self.assertEqual(sample["action_query_prev_token_ids"][-1], codec.model_token_ids[1])

    def test_pointmaze_bin_tokenization_does_not_add_aqt_metadata(self):
        tokenizer = DummyTokenizer(vocab_size=8)
        config = {
            "action_token_mode": "bin",
            "action_num_bins": 3,
            "action_bin_min": -1.0,
            "action_bin_max": 1.0,
            "new_token": True,
            "action_dim": 2,
            "max_length": 32,
        }
        codec = get_action_bin_codec(tokenizer, config, ensure_registered=True)
        old_tokenizer = pointmaze_dataset._POINTMAZE_WORKER_TOKENIZER
        old_codec = pointmaze_dataset._POINTMAZE_WORKER_ACTION_CODEC
        try:
            pointmaze_dataset._POINTMAZE_WORKER_TOKENIZER = tokenizer
            pointmaze_dataset._POINTMAZE_WORKER_ACTION_CODEC = codec
            sample = pointmaze_dataset._tokenize_pointmaze_sample(
                "prompt",
                codec.display_text_for_bins([1, 2]),
                config,
                expected_action_token_ids=codec.token_ids_for_bins([1, 2]),
                expected_action_bin_indices=[1, 2],
            )
        finally:
            pointmaze_dataset._POINTMAZE_WORKER_TOKENIZER = old_tokenizer
            pointmaze_dataset._POINTMAZE_WORKER_ACTION_CODEC = old_codec

        self.assertNotIn("action_query_mask", sample)
        self.assertNotIn("action_query_offsets", sample)
        self.assertNotIn("position_ids", sample)
        self.assertEqual(
            [label for label in sample["action_bin_labels"] if label >= 0],
            [1, 2],
        )

    def test_pointmaze_cache_signature_omits_source_hashes(self):
        config = pointmaze_dataset.PointMazeBuildConfig(
            variant="large",
            split="train",
            tokenizer_name_or_path="dummy-tokenizer",
            max_length=32,
            num_workers=1,
            cache_dir=None,
            max_data_num=None,
            dataset_partition_count=1,
            dataset_partition_index=None,
            prompt_template_count=1,
            prompt_templete_index=["bin_full_sensing"],
            train_data_ratio=0.9,
            episode_keep_num=None,
            balance_variant_episode_count=False,
            balanced_train_episode_count=None,
            sampling_seed=0,
            family_data_config=None,
            local_dataset_root=None,
            history_num=0,
            history_stride=1,
            wall_sensing_version="v3",
            map_sensing_boundary_risk_threshold=0.10,
            action_token_mode="bin",
            action_num_bins=3,
            action_bin_min=-1.0,
            action_bin_max=1.0,
            new_token=True,
            action_dim=2,
            mtp_k=None,
            action_token_schema_hash="dummy",
        )

        bin_payload = pointmaze_dataset.PointMazeDataset._cache_signature_payload(config)
        mtp_payload = pointmaze_dataset.PointMazeDataset._cache_signature_payload(
            replace(config, action_token_mode="mtp_bin", mtp_k=1)
        )

        self.assertNotIn("source_hashes", bin_payload)
        self.assertNotIn("source_hashes", mtp_payload)

    def test_eval_uses_mtp_quadratic_forward_instead_of_generate(self):
        tokenizer = DummyTokenizer(vocab_size=8)
        config = {
            "action_token_mode": "mtp_bin",
            "action_num_bins": 3,
            "action_bin_min": -1.0,
            "action_bin_max": 1.0,
            "new_token": True,
            "action_dim": 2,
            "mtp_k": 1,
            "mtp_quadratic_decoding": True,
            "max_length": 32,
            "action_sampling": False,
        }
        codec = get_action_bin_codec(tokenizer, config, ensure_registered=True)
        model = DummyMTPBinModel(
            vocab_size=len(tokenizer),
            bin_token_ids=list(codec.model_token_ids),
        )
        action_context = build_action_rollout_context(
            config=config,
            tokenizer=tokenizer,
            action_dim=2,
            collect_bin_probabilities=True,
        )

        result = generate_valid_action(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            formatter=DummyFormatter(),
            prompt="prompt",
            config=config,
            action_context=action_context,
            action_shape=(2,),
            action_dim=2,
            parse_retry_limit=3,
        )

        np.testing.assert_allclose(result.action, np.array([1.0, -1.0], dtype=np.float32))
        self.assertEqual(result.generated_attempts, ["<act_02><act_00>"])
        self.assertEqual(result.executed_action_text, "<act_02><act_00>")
        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(result.generation_count, 1)
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(len(result.generated_probability_logs), 1)

    def test_eval_can_trust_direct_mtp_output_without_quadratic_decoding(self):
        tokenizer = DummyTokenizer(vocab_size=8)
        config = {
            "action_token_mode": "mtp_bin",
            "action_num_bins": 3,
            "action_bin_min": -1.0,
            "action_bin_max": 1.0,
            "new_token": True,
            "action_dim": 2,
            "mtp_k": 1,
            "mtp_quadratic_decoding": False,
            "max_length": 32,
            "action_sampling": False,
        }
        codec = get_action_bin_codec(tokenizer, config, ensure_registered=True)
        model = DummyMTPBinModel(
            vocab_size=len(tokenizer),
            bin_token_ids=list(codec.model_token_ids),
        )
        action_context = build_action_rollout_context(
            config=config,
            tokenizer=tokenizer,
            action_dim=2,
            collect_bin_probabilities=True,
        )

        result = generate_valid_action(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            formatter=DummyFormatter(),
            prompt="prompt",
            config=config,
            action_context=action_context,
            action_shape=(2,),
            action_dim=2,
            parse_retry_limit=3,
        )

        np.testing.assert_allclose(result.action, np.array([1.0, -1.0], dtype=np.float32))
        self.assertEqual(result.generated_attempts, ["<act_02><act_00>"])
        self.assertEqual(result.executed_action_text, "<act_02><act_00>")
        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(result.generation_count, 1)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(result.generated_probability_logs), 1)


if __name__ == "__main__":
    unittest.main()
