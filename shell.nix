with import <nixpkgs> {};

mkShell {
  buildInputs = [
    nodejs
    electron

    gtk3
    nss
    nspr
    atk
    at-spi2-atk
    cairo
    pango
    glib

    libx11
    libxcomposite
    libxcursor
    libxdamage
    libxext
    libxfixes
    libxi
    libxrandr
    libxrender
    libxtst
    libxcb

    libxkbcommon
    dbus
    expat
    mesa
    alsa-lib

    cups

	python3
	python3Packages.torch
	python3Packages.numpy
	python3Packages.tokenizers
  ];

  shellHook = ''
    export ELECTRON_OVERRIDE_DIST_PATH=${electron}/bin
  '';
}
