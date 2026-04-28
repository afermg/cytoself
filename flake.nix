{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    pynng-flake.url = "github:afermg/pynng";
    pynng-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      with pkgs;
      rec {
        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              packages.nahual
              pp.torch
              pp.torchvision
              pp.numpy
              pp.pillow
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              export PYTHONPATH=${self}:$PYTHONPATH
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/cytoself.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        packages = {
          nahual = pkgs.python3.pkgs.callPackage ./nix/nahual.nix {
            pynng = inputs.pynng-flake.packages.${system}.pynng;
          };
        };

        devShells = {
          default =
            let
              python_with_pkgs = python3.withPackages (pp: [
                packages.nahual
                pp.torch
                pp.torchvision
                pp.numpy
                pp.pillow
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
              ];
              shellHook = ''
                export PYTHONPATH=${self}:${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
