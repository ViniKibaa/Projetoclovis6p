"""Geração de dataset sintético, leitura de CSV e divisão treino/val/teste.

O gerador combina frases-base escritas à mão (12 células: 4 categorias x
3 urgências, 10 frases por célula) com geração composicional e cria variações (saudações, fechamentos,
local, erros de digitação, ausência de acentos). Cada frase-base recebe um
identificador de ``grupo``: a divisão treino/val/teste é feita *por grupo*,
de modo que o teste contém frases que a rede nunca viu — avaliação bem mais
honesta do que sortear variações da mesma frase para treino e teste.
"""

import csv
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import CATEGORIAS_PADRAO, URGENCIAS_PADRAO

CATALOGO: dict[str, dict[str, list[str]]] = {
    "Hardware": {
        "Alto": [
            "o servidor principal desligou sozinho e todos os sistemas estão fora do ar",
            "o computador do caixa não liga e a loja está parada sem conseguir vender",
            "sai cheiro de queimado do nobreak da sala de servidores",
            "o disco do servidor de arquivos falhou e ninguém acessa os documentos da empresa",
            "a impressora de etiquetas da expedição quebrou e nenhum pedido sai hoje",
            "o notebook do diretor caiu e a tela quebrou antes da reunião com o cliente em uma hora",
            "o rack está superaquecendo e desligando equipamentos da produção",
            "vários computadores do laboratório não ligam depois da queda de energia",
            "o storage de backup emitiu alarme de disco danificado e os dados da empresa estão em risco",
            "a catraca de entrada travou e nenhum funcionário consegue entrar no prédio",
        ],
        "Médio": [
            "meu notebook está superaquecendo e desligando durante as videochamadas",
            "o monitor fica piscando de vez em quando e atrapalha meu trabalho",
            "a impressora do andar trava o papel com frequência",
            "o teclado perdeu algumas teclas e preciso de outro para trabalhar direito",
            "o computador está muito lento e demora minutos para abrir qualquer programa",
            "o mouse sem fio desconecta toda hora",
            "a bateria do notebook dura menos de vinte minutos",
            "o headset não funciona nas reuniões e uso o celular como alternativa",
            "o scanner da recepção deixa listras nos documentos digitalizados",
            "a webcam do meu computador não é reconhecida e preciso dela para as reuniões da semana",
        ],
        "Baixo": [
            "gostaria de solicitar um suporte de monitor para a minha mesa",
            "o adesivo do patrimônio do meu computador descolou",
            "queria trocar meu mouse por um modelo mais confortável quando possível",
            "posso pegar emprestado um adaptador HDMI na semana que vem",
            "a tampa lateral do gabinete está meio solta mas funciona normalmente",
            "preciso de um cabo de carregador reserva para deixar na mesa",
            "dúvida sobre a forma correta de limpar a tela do notebook",
            "solicito um mousepad novo quando houver estoque",
            "tem como trocar a cor da capa do notebook corporativo no próximo ciclo de compras",
            "pergunta sobre a garantia do meu monitor novo",
        ],
    },
    "Redes": {
        "Alto": [
            "a internet caiu em todo o prédio e ninguém consegue trabalhar",
            "a VPN está fora do ar e toda a equipe remota está parada",
            "o wi-fi da empresa inteira não conecta desde cedo",
            "o link principal está down e o sistema de vendas ficou offline em todas as lojas",
            "o firewall parou de responder e as filiais perderam a conexão com a matriz",
            "o servidor DNS não resolve nomes e todos os sistemas internos estão inacessíveis",
            "ataque de negação de serviço está derrubando o site do e-commerce",
            "o switch central travou e o andar inteiro está sem rede",
            "o roteador da matriz reiniciou em loop e as filiais estão sem comunicação",
            "a rede da fábrica caiu e as máquinas da linha de produção pararam",
        ],
        "Médio": [
            "a internet está muito lenta na minha sala desde ontem",
            "o wi-fi cai a cada dez minutos no meu setor",
            "não consigo acessar a pasta compartilhada da rede do financeiro",
            "as videochamadas ficam travando por causa da conexão instável",
            "a VPN conecta mas fica lenta e desconecta depois de um tempo",
            "o ponto de rede da minha mesa não funciona e estou usando wi-fi",
            "não consigo conectar o notebook na rede da sala de reunião",
            "o site da intranet demora muito para carregar",
            "o ping para o servidor de arquivos oscila bastante no meu andar",
            "a rede cabeada da minha baia perde pacotes e a transferência de arquivos falha",
        ],
        "Baixo": [
            "gostaria de saber a senha do wi-fi para visitantes da semana que vem",
            "dúvida sobre qual rede devo usar no meu celular pessoal",
            "queria solicitar um ponto de rede extra para a mesa nova no próximo mês",
            "como configuro a impressora de rede no meu notebook novo quando der",
            "pergunta sobre o horário da manutenção programada da rede",
            "o nome do wi-fi dos visitantes está com erro de digitação",
            "gostaria de entender a diferença entre as redes wi-fi da empresa",
            "posso pedir um cabo de rede mais longo para minha estação",
            "existe um mapa com a cobertura do wi-fi nos andares",
            "quero saber qual é a velocidade contratada do nosso link de internet",
        ],
    },
    "Software": {
        "Alto": [
            "o sistema de faturamento está fora do ar e não conseguimos emitir notas fiscais",
            "o ERP apresenta erro crítico e todos os usuários perderam o acesso aos pedidos",
            "o banco de dados corrompeu e a aplicação de produção parou",
            "a atualização do sistema apagou os dados dos clientes",
            "um ransomware criptografou os arquivos de vários computadores e exige resgate",
            "o sistema de ponto eletrônico travou e ninguém consegue registrar a entrada",
            "o aplicativo de vendas fecha sozinho para todos os vendedores em campo",
            "o servidor de e-mail parou e a empresa inteira está sem e-mail",
            "o sistema de pagamentos retorna erro em todas as transações e os clientes não conseguem comprar",
            "a nova versão do sistema derrubou o atendimento e as filas estão paradas",
        ],
        "Médio": [
            "o Excel fecha sozinho quando abro a planilha de relatórios",
            "o programa de edição de vídeo trava toda vez que exporto",
            "o navegador está lento e mostra erro ao abrir o sistema de chamados",
            "o Outlook não sincroniza meus e-mails desde ontem",
            "a atualização do Windows falhou e o computador fica reiniciando em loop",
            "o antivírus acusou um arquivo suspeito no meu computador",
            "o sistema trava ao gerar o relatório mensal e perco o que estava fazendo",
            "o Teams não abre a câmera nas reuniões",
            "o plugin do navegador deixou de funcionar e não consigo assinar os documentos",
            "o aplicativo de contabilidade apresenta erro ao importar o arquivo do banco",
        ],
        "Baixo": [
            "gostaria de instalar um editor de PDF no meu computador quando possível",
            "dúvida sobre como alterar a assinatura do e-mail",
            "pode atualizar a versão do navegador na próxima manutenção",
            "queria um tutorial de como usar as planilhas compartilhadas",
            "solicito a instalação de um programa de captura de tela",
            "o tema escuro do sistema interno poderia ser configurado",
            "como altero o idioma do aplicativo para português",
            "sugestão de melhoria: incluir um botão de exportar na tela de relatórios",
            "existe alguma licença disponível do editor de imagens para testar",
            "gostaria de saber como organizar as pastas do meu e-mail",
        ],
    },
    "Acessos/Senhas": {
        "Alto": [
            "minha conta foi invadida e estão enviando e-mails em meu nome",
            "o diretório de usuários travou e ninguém consegue fazer login em nenhum sistema",
            "todos os usuários do financeiro estão bloqueados e o fechamento do mês vence hoje",
            "suspeita de vazamento da senha do administrador do servidor",
            "o login único está fora do ar e nenhum colaborador entra nos sistemas",
            "foi detectado acesso indevido a uma conta de administrador agora mesmo",
            "o token de autenticação em dois fatores parou e a equipe inteira está sem acesso",
            "um ex-funcionário ainda acessa o sistema de clientes e precisa ser bloqueado imediatamente",
            "alguém tentou entrar com minha senha de outro país e a conta de diretoria está exposta",
            "o servidor de autenticação caiu e todo o call center está sem conseguir logar",
        ],
        "Médio": [
            "esqueci minha senha e preciso acessar o sistema para entregar um relatório hoje",
            "minha conta está bloqueada depois de várias tentativas de login",
            "não recebo o código de verificação em dois fatores no celular",
            "preciso de acesso à pasta do projeto novo para continuar o meu trabalho",
            "minha senha expirou e o portal não deixa cadastrar outra",
            "perdi o acesso ao sistema de compras depois da troca de cargo",
            "o acesso à VPN foi negado para o meu usuário",
            "a permissão de edição da planilha compartilhada foi removida e preciso dela",
            "o aplicativo autenticador foi apagado do meu celular e não consigo entrar no e-mail",
            "meu usuário não aparece no grupo do departamento e não vejo os arquivos da equipe",
        ],
        "Baixo": [
            "gostaria de saber qual é a política de troca de senhas",
            "dúvida sobre como criar uma senha mais forte",
            "quero solicitar acesso de leitura a um relatório que consulto de vez em quando",
            "como cadastro um novo celular na autenticação de dois fatores quando trocar de aparelho",
            "pergunta sobre o prazo de validade das senhas da empresa",
            "poderia incluir meu nome na lista de acesso ao evento do mês que vem",
            "gostaria de atualizar o e-mail de recuperação da minha conta",
            "solicito um cofre de senhas para organizar meus acessos quando possível",
            "existe um guia sobre como funcionam os perfis de acesso",
            "queria entender por que o sistema pede troca de senha a cada noventa dias",
        ],
    },
}


# --- Geração composicional: sujeito + sintoma + cláusula de impacto -----------
# A categoria vem do sujeito/sintoma; a urgência, principalmente do impacto
# descrito (quantas pessoas, prazo, risco). Isso dá diversidade lexical bem
# maior do que apenas frases fixas.
SUJEITOS: dict[str, list[str]] = {
    "Hardware": ["o notebook", "o monitor", "a impressora", "o teclado", "o mouse", "o headset",
                 "o servidor de arquivos", "o nobreak", "o projetor da sala de reunião", "o scanner",
                 "a fonte do computador", "o HD externo", "a webcam", "o desktop da recepção"],
    "Redes": ["a internet", "o wi-fi", "a VPN", "a rede cabeada", "o link da filial",
              "o ponto de rede da minha mesa", "o acesso à intranet", "o servidor DNS", "o firewall",
              "a conexão com o servidor", "o roteador", "o switch do andar"],
    "Software": ["o Excel", "o Outlook", "o sistema de faturamento", "o ERP", "o Teams", "o navegador",
                 "o aplicativo de vendas", "o antivírus", "o sistema de chamados",
                 "o programa de contabilidade", "o Windows", "o editor de planilhas"],
    "Acessos/Senhas": ["minha senha", "minha conta", "meu acesso ao sistema", "o login único",
                       "o código de verificação em dois fatores", "o acesso ao e-mail",
                       "a permissão da pasta compartilhada", "o cadastro no portal",
                       "a conta do administrador", "o acesso ao sistema de compras", "meu usuário"],
}
SINTOMAS: dict[str, list[str]] = {
    "Hardware": ["não liga", "parou de funcionar", "esquenta demais e desliga", "faz um barulho estranho",
                 "deixou de responder", "apresenta defeito", "falhou de repente", "não funciona",
                 "deixou de funcionar"],
    "Redes": ["caiu", "cai toda hora", "oscila bastante", "não conecta", "perdeu a conexão",
              "apresenta falha de comunicação", "ficou instável", "não funciona", "parou de funcionar"],
    "Software": ["trava ao abrir", "fecha sozinho", "mostra uma mensagem de erro", "não abre",
                 "parou de funcionar depois da atualização", "não salva os dados", "apresenta erro ao iniciar",
                 "não funciona", "deixou de funcionar"],
    "Acessos/Senhas": ["expirou", "foi bloqueado", "não funciona", "deixou de funcionar", "foi negado",
                       "falhou na autenticação", "apresenta erro ao entrar", "parou de funcionar"],
}
IMPACTOS: dict[str, list[str]] = {
    "Alto": [
        "e toda a empresa está parada", "e todos os setores estão sem conseguir trabalhar",
        "e estamos perdendo vendas agora", "e a operação inteira parou", "e os clientes estão sendo afetados, é urgente",
        "e o prejuízo aumenta a cada minuto", "e ninguém consegue trabalhar", "e a diretoria cobra solução imediata",
        "e há risco para os dados da empresa", "e todas as filiais estão sem atendimento",
    ],
    "Médio": [
        "e isso atrapalha meu trabalho hoje", "e preciso resolver até o fim do dia",
        "e só eu estou com esse problema, mas tenho um relatório para entregar",
        "e consigo trabalhar apenas de forma limitada", "mas ainda dá para usar com dificuldade",
        "e isso está atrasando minhas tarefas", "e preciso de uma solução ainda esta semana",
        "e afeta duas pessoas da minha equipe", "e estou usando um contorno provisório",
    ],
    "Baixo": [
        "mas não tem pressa", "e pode ser resolvido na próxima semana", "e dá para ver quando houver uma oportunidade",
        "sem urgência", "e não está atrapalhando meu trabalho", "só estou avisando para constar",
        "e pode ser tratado na próxima manutenção", "mas tenho outra opção funcionando bem",
        "e fico no aguardo sem prazo definido",
    ],
}

ABERTURAS = ("", "", "", "Bom dia. ", "Boa tarde, ", "Olá, ", "Oi, ", "Preciso de ajuda: ", "Pessoal, ")
FECHAMENTOS = ("", "", "", " Podem verificar?", " Obrigado.", " Aguardo retorno.", " Qualquer novidade me avisem.")
LOCAIS = ("", "", "", " Estou no 2º andar.", " Sala 204.", " Setor comercial.", " Setor administrativo.", " Filial centro.")


def _sem_acentos(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _typo(s: str, rng: random.Random) -> str:
    palavras = s.split(" ")
    candidatas = [i for i, p in enumerate(palavras) if len(p) > 4 and p.isalpha()]
    if not candidatas:
        return s
    i = rng.choice(candidatas)
    p = list(palavras[i])
    j = rng.randrange(1, len(p) - 2)
    p[j], p[j + 1] = p[j + 1], p[j]
    palavras[i] = "".join(p)
    return " ".join(palavras)


def _variar(base: str, rng: random.Random) -> str:
    texto = rng.choice(ABERTURAS) + base
    if texto[0].isalpha() and rng.random() < 0.8:
        texto = texto[0].upper() + texto[1:]
    texto += rng.choice(("", ".", "."))
    texto += rng.choice(FECHAMENTOS) + rng.choice(LOCAIS)
    if rng.random() < 0.12:
        texto = _typo(texto, rng)
    if rng.random() < 0.15:
        texto = _sem_acentos(texto)
    if rng.random() < 0.10:
        texto = texto.lower()
    return " ".join(texto.split())


def gerar_dataset(n: int = 480, seed: int = 7) -> list[dict]:
    """Gera ~``n`` linhas (texto, categoria, urgencia, grupo), balanceadas.

    Metade vem de frases escritas à mão (com variações) e metade da geração
    composicional (sujeito + sintoma + impacto).
    """
    rng = random.Random(seed)
    celulas = [(c, u) for c in CATEGORIAS_PADRAO for u in URGENCIAS_PADRAO]
    por_celula = max(2, n // len(celulas))
    n_manual = por_celula // 2
    linhas: list[dict] = []
    for cat, urg in celulas:
        bases = CATALOGO[cat][urg]
        vistos: set[str] = set()
        for g, base in enumerate(bases):
            alvo = n_manual // len(bases) + (1 if g < n_manual % len(bases) else 0)
            tentativas = 0
            while alvo > 0 and tentativas < 200:
                tentativas += 1
                texto = _variar(base, rng)
                if texto in vistos:
                    continue
                vistos.add(texto)
                alvo -= 1
                linhas.append({"texto": texto, "categoria": cat, "urgencia": urg, "grupo": f"{cat}|{urg}|m{g}"})
        restante, tentativas = por_celula - n_manual, 0
        while restante > 0 and tentativas < 2000:
            tentativas += 1
            suj, sint = rng.choice(SUJEITOS[cat]), rng.choice(SINTOMAS[cat])
            base = f"{suj} {sint} {rng.choice(IMPACTOS[urg])}"
            texto = _variar(base, rng)
            if texto in vistos:
                continue
            vistos.add(texto)
            restante -= 1
            linhas.append({"texto": texto, "categoria": cat, "urgencia": urg, "grupo": f"{cat}|{urg}|c:{suj}|{sint}"})
    rng.shuffle(linhas)
    for i, l in enumerate(linhas, 1):
        l["id"] = i
    return linhas


def salvar_csv(linhas: list[dict], caminho) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "texto", "categoria", "urgencia", "grupo"])
        w.writeheader()
        w.writerows(linhas)


def carregar_csv(caminho) -> dict:
    """Lê CSV com colunas ``texto,categoria,urgencia`` (``grupo`` é opcional)."""
    with open(caminho, encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        faltando = {"texto", "categoria", "urgencia"} - set(leitor.fieldnames or [])
        if faltando:
            raise ValueError(f"CSV sem as colunas obrigatórias: {sorted(faltando)}")
        textos, cats, urgs, grupos = [], [], [], []
        for l in leitor:
            t = (l["texto"] or "").strip()
            c = (l["categoria"] or "").strip()
            u = (l["urgencia"] or "").strip()
            if not (t and c and u):
                continue
            textos.append(t)
            cats.append(c)
            urgs.append(u)
            grupos.append((l.get("grupo") or "").strip())
    if len(textos) < 30:
        raise ValueError("Poucos exemplos válidos no CSV (mínimo 30).")
    tem_grupo = all(grupos)
    return {
        "textos": textos,
        "categorias": cats,
        "urgencias": urgs,
        "grupos": grupos if tem_grupo else None,
    }


def ordenar_rotulos(rotulos, preferida) -> list[str]:
    """Mantém a ordem semântica padrão (ex.: Baixo<Médio<Alto) quando possível."""
    unicos = set(rotulos)
    if unicos <= set(preferida):
        return [r for r in preferida if r in unicos]
    return sorted(unicos)


def dividir(cats, urgs, grupos=None, val=0.15, teste=0.15, seed=42) -> dict[str, np.ndarray]:
    """Divide em treino/val/teste estratificando por (categoria, urgência).

    Se ``grupos`` for informado, a divisão é por grupo (frases-base inéditas
    no teste); caso contrário, é uma divisão aleatória estratificada.
    """
    rng = random.Random(seed)
    estratos: dict[tuple, list[int]] = defaultdict(list)
    for i, chave in enumerate(zip(cats, urgs)):
        estratos[chave].append(i)
    out: dict[str, list[int]] = {"treino": [], "val": [], "teste": []}
    for chave in sorted(estratos):
        idxs = estratos[chave]
        if grupos is not None:
            por_grupo: dict[str, list[int]] = defaultdict(list)
            for i in idxs:
                por_grupo[grupos[i]].append(i)
            unidades = [por_grupo[g] for g in sorted(por_grupo)]
        else:
            unidades = [[i] for i in idxs]
        rng.shuffle(unidades)
        n = len(unidades)
        if n < 3:
            out["treino"] += [i for u in unidades for i in u]
            continue
        n_te = max(1, round(teste * n))
        n_va = max(1, round(val * n))
        out["teste"] += [i for u in unidades[:n_te] for i in u]
        out["val"] += [i for u in unidades[n_te : n_te + n_va] for i in u]
        out["treino"] += [i for u in unidades[n_te + n_va :] for i in u]
    return {k: np.array(sorted(v), dtype=int) for k, v in out.items()}
