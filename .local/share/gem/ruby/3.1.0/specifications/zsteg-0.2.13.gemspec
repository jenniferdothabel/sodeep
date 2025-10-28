# -*- encoding: utf-8 -*-
# stub: zsteg 0.2.13 ruby lib

Gem::Specification.new do |s|
  s.name = "zsteg".freeze
  s.version = "0.2.13".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Andrey \"Zed\" Zaikin".freeze]
  s.date = "2023-02-19"
  s.email = "zed.0xff@gmail.com".freeze
  s.executables = ["zsteg".freeze, "zsteg-mask".freeze, "zsteg-reflow".freeze]
  s.extra_rdoc_files = ["README.md".freeze, "README.md.tpl".freeze, "TODO".freeze]
  s.files = ["README.md".freeze, "README.md.tpl".freeze, "TODO".freeze, "bin/zsteg".freeze, "bin/zsteg-mask".freeze, "bin/zsteg-reflow".freeze]
  s.homepage = "http://github.com/zed-0xff/zsteg".freeze
  s.licenses = ["MIT".freeze]
  s.rubygems_version = "3.5.9".freeze
  s.summary = "Detect stegano-hidden data in PNG & BMP files.".freeze

  s.installed_by_version = "3.5.9".freeze if s.respond_to? :installed_by_version

  s.specification_version = 4

  s.add_runtime_dependency(%q<zpng>.freeze, [">= 0.4.5".freeze])
  s.add_runtime_dependency(%q<iostruct>.freeze, [">= 0.0.5".freeze])
  s.add_runtime_dependency(%q<prime>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<rspec>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<juwelier>.freeze, [">= 0".freeze])
end
