import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dataset import build_loaders
from utils_vn_connect import add_ppa_virtual_nodes
from vn_cache import CachedVNDataset


class GraphDataset(Dataset):
    num_tasks = 1
    num_classes = 37

    def __init__(self, root):
        self.processed_dir = root
        self.processed_paths = [str(Path(root) / 'graphs.pt')]
        Path(self.processed_paths[0]).write_bytes(b'dataset-v1')
        # Triangle, disconnected graph, singleton: includes empty VN attachments
        # and an empty VN edge_index under the first-bag assignment policy.
        self.graphs = []
        for n, pairs in [(3, [(0, 1), (1, 2), (2, 0)]), (4, [(0, 1)]), (1, [])]:
            edges = pairs + [(v, u) for u, v in pairs]
            edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t()
            self.graphs.append(Data(
                num_nodes=n, edge_index=edge_index,
                edge_attr=torch.ones(len(edges), 7), y=torch.tensor([[0]]),
            ))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[int(index)]

    def get_idx_split(self):
        return {name: torch.tensor([i]) for i, name in enumerate(('train', 'valid', 'test'))}


class VNCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset = GraphDataset(self.temp.name)

    def test_reuse_across_epochs_and_instances_preserves_batched_graphs(self):
        expected = Batch.from_data_list([add_ppa_virtual_nodes(g) for g in self.dataset])
        with patch('vn_cache.add_ppa_virtual_nodes', wraps=add_ppa_virtual_nodes) as transform:
            cached = CachedVNDataset(self.dataset)
            first = Batch.from_data_list([cached[i] for i in range(len(cached))])
            self.assertEqual(transform.call_count, 3)
            reused = CachedVNDataset(self.dataset)
            second = Batch.from_data_list([reused[i] for i in range(len(reused))])
            self.assertEqual(transform.call_count, 3)
        for key in expected.keys():
            for actual in (first, second):
                if isinstance(expected[key], torch.Tensor):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                else:
                    self.assertEqual(actual[key], expected[key])
        torch.testing.assert_close(second.vn_batch[second.node2vn], second.batch)
        self.assertTrue(all('node2vn' not in g for g in self.dataset))
        fields = torch.load(next(cached.cache_dir.rglob('*.pt')), weights_only=True)
        self.assertEqual(set(fields), {'vn_edge_index', 'node2vn', 'vn_batch'})

    def test_data_and_implementation_changes_select_new_cache(self):
        original = CachedVNDataset(self.dataset).cache_dir
        Path(self.dataset.processed_paths[0]).write_bytes(b'updated-dataset-v2')
        updated = CachedVNDataset(self.dataset).cache_dir
        self.assertNotEqual(original, updated)
        implementation = Path(self.temp.name) / 'new_implementation.py'
        implementation.write_text('# updated VN preprocessing')
        with patch('vn_cache.utils_vn_connect.__file__', str(implementation)):
            self.assertNotEqual(updated, CachedVNDataset(self.dataset).cache_dir)

    def test_restarted_workers_reuse_disk_cache(self):
        cached = CachedVNDataset(self.dataset)
        first = list(DataLoader(cached, batch_size=2, num_workers=2))
        files = {p: p.stat().st_mtime_ns for p in cached.cache_dir.rglob('*.pt')}
        self.assertEqual(len(files), len(cached))
        with patch('vn_cache.add_ppa_virtual_nodes', side_effect=AssertionError('cache miss')):
            second = list(DataLoader(CachedVNDataset(self.dataset), batch_size=2, num_workers=2))
        self.assertEqual(files, {p: p.stat().st_mtime_ns for p in files})
        for before, after in zip(first, second):
            torch.testing.assert_close(before.node2vn, after.node2vn)
            torch.testing.assert_close(before.vn_edge_index, after.vn_edge_index)

    def test_failed_write_does_not_publish_partial_cache(self):
        cached = CachedVNDataset(self.dataset)
        with patch('vn_cache.torch.save', side_effect=OSError('write failed')):
            with self.assertRaisesRegex(OSError, 'write failed'):
                cached[0]
        self.assertEqual(list(cached.cache_dir.rglob('*.pt')), [])
        self.assertEqual(list(cached.cache_dir.rglob('*.tmp')), [])
        self.assertEqual(cached[0].num_nodes, 3)

    def test_all_loaders_use_cache(self):
        cfg = {'data': {'dataset_name': 'ogbg-ppa', 'dir': self.temp.name},
               'train': {'batch_size': 2, 'num_workers': 0}}
        with patch('dataset.PygGraphPropPredDataset', return_value=self.dataset) as factory:
            loaders = build_loaders(cfg)
        self.assertNotIn('transform', factory.call_args.kwargs)
        self.assertEqual(loaders[3:], (1, 37))
        with patch('vn_cache.add_ppa_virtual_nodes', wraps=add_ppa_virtual_nodes) as transform:
            for _ in range(2):
                for loader in loaders[:3]:
                    self.assertEqual(sum(batch.num_graphs for batch in loader), 1)
            self.assertEqual(transform.call_count, 3)


if __name__ == '__main__':
    unittest.main()
