{
  description = "PINNs";

  inputs = {
    # can stay on two channels if you really need both
    nixpkgs-unstable.url = "github:NixOS/nixpkgs?ref=nixos-unstable";
    nixpkgs.url          = "github:NixOS/nixpkgs?ref=nixos-25.05";
  };

  outputs = { self, nixpkgs, nixpkgs-unstable }:
  let
    system = "x86_64-linux";

    # ──────────────── Channels ────────────────
    pkgs-unstable = import nixpkgs-unstable {
      inherit system;
      pure   = true;
      config = { allowUnfree = true; };
    };

    pkgs = import nixpkgs {
      inherit system;
      pure   = true;
      config = { allowUnfree = true; };
    };

    # ──────────────── Python env ────────────────
    pythonEnv = pkgs.python313.withPackages (ps: with ps; [
      torch numpy ipython scipy einops utils matplotlib
      fenics-dolfinx tqdm tensorly opt-einsum
    ]);

    # ──────────────── FHS env (ex-shell.nix) ────────────────
    #
    # buildFHSUserEnv crée un root FS compatible FHS, l’attribut `.env`
    # est le derivation à exposer comme devShell (cf. doc) :contentReference[oaicite:0]{index=0}
    #
    fhsEnv = pkgs.buildFHSEnv {
      name = "pipzone";

      # Les paquets disponibles dans le chroot
      targetPkgs = pkgs: with pkgs; [
        libgcc binutils coreutils
        pythonEnv          # on y met directement notre python custom
      ];

      # Les variables d’environnement reprises de shell.nix
      profile = ''
        # Lignes originales de shell.nix
        export LIBRARY_PATH=/usr/lib:/usr/lib64:$LIBRARY_PATH
        # export LIBRARY_PATH=${pkgs.libgcc}/lib
      '';

      runScript = "bash";
    };

  in {
    # ──────────────── Exports ────────────────

    # 1. Le package principal (pour nix run .#default)
    packages.${system}.default = pythonEnv;

    # 2. Deux environnements de dev :
    #    • default  : FHS (hérite de l’ancien shell.nix)
    #    • pure     : mkShell 100 % Nix, si besoin
    devShells.${system} = {
      # FHS env -> `nix develop` (ou `nix develop .#default`)
      default = fhsEnv.env;

      # Ancien mkShell gardé pour les cas où FHS n’est pas nécessaire
      pure = pkgs.mkShell {
        buildInputs = [ pkgs.bash pythonEnv ];

        # répète ta ligne LD_LIBRARY_PATH d’origine (corrigée)
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.libz
        ];
      };
    };
  };
}