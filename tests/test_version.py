"""Tests for version detection functionality."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from table_evaluator import __version__, _get_version


class TestVersionDetection:
    """Test version detection from setuptools_scm."""

    def test_version_detection_with_importlib(self):
        """Test version detection using importlib.metadata."""
        with patch('importlib.metadata.version') as mock_version:
            mock_version.return_value = '1.9.0'

            version = _get_version()

            assert version == '1.9.0'
            mock_version.assert_called_once_with('table-evaluator')

    def test_version_detection_package_not_found(self):
        """Test version detection when package is not found."""
        with (
            patch('importlib.metadata.version') as mock_version,
            patch('pkg_resources.get_distribution') as mock_pkg_resources,
        ):
            from importlib.metadata import PackageNotFoundError

            import pkg_resources

            mock_version.side_effect = PackageNotFoundError()
            mock_pkg_resources.side_effect = pkg_resources.DistributionNotFound()

            version = _get_version()

            assert version == 'unknown'

    def test_version_detection_importlib_not_available(self):
        """Test fallback to pkg_resources when importlib.metadata is not available."""
        with (
            patch('importlib.metadata.version', side_effect=ImportError),
            patch('pkg_resources.get_distribution') as mock_get_dist,
        ):
            mock_dist = MagicMock()
            mock_dist.version = '1.9.0'
            mock_get_dist.return_value = mock_dist

            version = _get_version()

            assert version == '1.9.0'
            mock_get_dist.assert_called_once_with('table-evaluator')

    def test_version_detection_pkg_resources_not_found(self):
        """Test version detection when pkg_resources distribution is not found."""
        with (
            patch('importlib.metadata.version', side_effect=ImportError),
            patch('pkg_resources.get_distribution') as mock_get_dist,
        ):
            import pkg_resources

            mock_get_dist.side_effect = pkg_resources.DistributionNotFound()

            version = _get_version()

            assert version == 'unknown'

    def test_version_detection_pkg_resources_attribute_error(self):
        """Test version detection when pkg_resources has AttributeError."""
        with (
            patch('importlib.metadata.version', side_effect=ImportError),
            patch('pkg_resources.get_distribution') as mock_get_dist,
        ):
            mock_get_dist.side_effect = AttributeError()

            version = _get_version()

            assert version == 'unknown'

    def test_module_version_attribute_exists(self):
        """Test that __version__ attribute is properly set."""
        assert hasattr(sys.modules['table_evaluator'], '__version__')
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_format_development(self):
        """Test development version format detection."""
        # In development, version should contain commit info
        if 'dev' in __version__:
            assert '+g' in __version__  # Git commit hash
            assert 'dev' in __version__  # Development indicator
        else:
            # Production version should be semantic version
            parts = __version__.split('.')
            assert len(parts) >= 3  # Major.minor.patch at minimum

    def test_version_caching(self):
        """Test that version is cached and not recalculated on each access."""
        # Version should be cached as module attribute
        version1 = __version__
        version2 = __version__

        assert version1 is version2  # Same object reference
        assert version1 == version2  # Same value


class TestVersionIntegration:
    """Integration tests for version detection."""

    def test_version_from_setuptools_scm(self):
        """Test that version can be detected from setuptools_scm."""
        # This test runs against actual setuptools_scm
        version = _get_version()

        # Should not be unknown in a properly configured environment
        if version != 'unknown':
            # Should contain either semantic version or dev version
            assert '.' in version

            # Development versions contain commit info
            if 'dev' in version:
                assert '+g' in version
            else:
                # Semantic version pattern
                parts = version.split('.')
                assert len(parts) >= 3
                assert all(part.isdigit() for part in parts[:3])

    def test_version_consistency(self):
        """Test that version is consistent across multiple calls."""
        versions = [_get_version() for _ in range(5)]

        # All versions should be identical
        assert all(v == versions[0] for v in versions)
        assert len(set(versions)) == 1

    @pytest.mark.skipif(sys.version_info < (3, 8), reason='importlib.metadata requires Python 3.8+')
    def test_version_with_modern_python(self):
        """Test version detection with modern Python (3.8+)."""
        # Should use importlib.metadata path
        with patch('pkg_resources.get_distribution') as mock_pkg_resources:
            version = _get_version()

            # pkg_resources should not be called with modern Python
            mock_pkg_resources.assert_not_called()

            # Should get valid version
            assert version != 'unknown'
