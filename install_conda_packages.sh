while read package; do
    conda install --yes "$package" || { echo "Failed to install $package with default channel, trying conda-forge..."; conda install --yes -c conda-forge "$package" || echo "Failed to install $package from conda-forge"; }
done < conda-packages.txt
